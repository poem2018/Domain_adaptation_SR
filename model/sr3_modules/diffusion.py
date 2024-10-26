import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

import random
import numpy as np
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from torchvision.transforms import functional as trans_fn
import torchvision.transforms as transforms
from PIL import Image



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        #opt, ##TODO
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        # super(GaussianDiffusion, self).__init__()  #TODO

        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional  

        

        ##TODO
        # self.opt = opt   

        # self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.data = {}
        self.SR = None
        self.scale = None

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):   
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):   #backward inference
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, x_add = None, ddrm=False):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        
        ####revise: add scale into time################
        add_scale_into_t = False
        if add_scale_into_t == True:
            scale = self.scale.to(dtype=torch.float32)
            noise_level = torch.cat((noise_level, scale), dim = 1)
        #######################################
        
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        # DDRM trick
        # import pdb;pdb.set_trace()
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        # print("!!!!",ddrm)
        if ddrm:
            bicubic_x = torch.nn.functional.interpolate(x_recon, scale_factor=0.25, mode='bicubic')
            # import pdb;pdb.set_trace()
            diff = torch.tensor((bicubic_x - x_add), requires_grad=True).clone().detach()
            norm = torch.norm(diff)

            
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            dps_weight = 0.01
            model_mean = model_mean - norm_grad * dps_weight

        
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, x_add = None, ddrm=False):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, x_add = x_add, ddrm = ddrm)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        
        # x_next_mean = x_next_mean - norm_grad * dps_weight

        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, ddrm =False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in   #LR_resize   lr
            if ddrm:
                x =  trans_fn.resize(x_in, 1024, Image.BICUBIC)   #hard code
                x_add = x_in 
            shape = x.shape
            # import pdb;pdb.set_trace()
            img = torch.randn(shape, device=device)  #HR_noise
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                if ddrm:
                    img = self.p_sample(img, i, condition_x=x, x_add = x_add, ddrm=True)
                else:
                    img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)   ##??
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, scale, continous=False, ddrm=False):
        self.scale = scale.unsqueeze(0)
        return self.p_sample_loop(x_in, continous, ddrm)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )
    

    def print_fig(self, x_in, save_name, noise=None):
        lr = x_in.squeeze()
        lr_n = (lr - lr.min()) / (lr.max() - lr.min())
        lr_array = np.moveaxis(lr_n.cpu().numpy()*255, 0, -1)
        lr_image = Image.fromarray(lr_array.astype(np.uint8))
        lr_image.save(save_name)

        return True

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR'] 
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)


        ####revise: add the scale into time################
        add_scale_into_t = False
        if add_scale_into_t == True:
            self.scale = x_in['scale'].unsqueeze(0)
            scale = self.scale.to(dtype=torch.float32)
            continuous_sqrt_alpha_cumprod = torch.cat((continuous_sqrt_alpha_cumprod, scale),  dim = 1)
        ###########################################


        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        self.SR = x_recon
        # self.data['HR'] = x_in['HR']
        # self.data['LR']  = x_in['LR']
        # self.data['SR']  = self.SR
        ###TODO
        

        ######revise: add consistency loss###########################3
        consistency_loss = True 
        if consistency_loss == True:
            loss_rec = self.loss_func(noise, x_recon)   ###need modify
            scale = float((1/x_in['scale'])[0]) 

            # 
            x_rec = self.predict_start_from_noise(x_noisy, t-1, x_recon)  ###??????

            x_rec_downsample = F.interpolate(x_rec, scale_factor=(scale, scale), mode='bicubic', align_corners=False)
            loss_con = self.loss_func(x_in['LR'], x_rec_downsample)     ###is it this way?
            alpha = 0.6
            loss  = alpha*loss_rec+(1-alpha)*loss_con
            # print("loss_rec: ", loss_rec, "loss_con: ", loss_con)

            if random.random() > 0.5:  
                self.print_fig(x_in['LR'][0], 'test_lr.png')
                self.print_fig(x_rec_downsample.detach()[0], 'test_fake_lr.png')
                self.print_fig(x_in['HR'][0], 'test_hr.png')
                self.print_fig(x_rec.detach()[0], 'test_fake_hr.png')

        if consistency_loss== False:
            if add_scale_into_t == True:
                loss = self.loss_func(noise, x_recon)/4
            else:
                loss = self.loss_func(noise, x_recon)   ###need modify
        return loss


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
