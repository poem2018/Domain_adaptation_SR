import sys
import os

# assert len(sys.argv) == 3, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

# assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *

## python tool_add_control.py -i ./experiments/val_8x_zssr_16_128_240215_030646/checkpoint/I970000_E108 -o ./finetune_model/I970000_E108 -c ./config/zssr_16_128.json
## python tool_add_control.py -i ./experiments/val_finetune_2x_256_512_240215_030756/checkpoint/I480000_E76 -o ./finetune_model/I480000_E76 -c ./config/zssr_256_512.json

#python tool_add_control.py -i ./experiments/2x_IXI_240424_234312/checkpoint/I1000000_E68 -o ./finetune_model/I1000000_E68  -c ./config/IXI_128_256.json
# python tool_add_control.py -i ./experiments/2x_coco_contin_240505_123440/checkpoint/I1990000_E82 -o ./finetune_model/I1990000_E82  -c ./config/coco_128_256.json
# python tool_add_control.py -i ./experiments/coco_IXI_240514_153653/checkpoint/I2410000_E143 -o ./finetune_model/I2410000_E143  -c ./config/IXI_64_128.json

##I don't need to do second control net for ixi-fastmri finetune????

# 4x coco-ixi
# python tool_add_control.py -i ./experiments/2x_coco_240527_203413/checkpoint/I850000_E35 -o ./finetune_new/2x/I850000_E35  -c ./config/coco_128_256.json
# python tool_add_control.py -i ./experiments/4x_coco_240527_074939/checkpoint/I1000000_E41 -o ./finetune_new/4x/I1000000_E41  -c ./config/coco_128_256.json
# python tool_add_control.py -i ./experiments/8x_coco_240527_075154/checkpoint/I1000000_E41 -o ./finetune_new/8x/I1000000_E41  -c ./config/coco_128_256.json

# python tool_add_control.py -i ./experiments/4x_coco_adamw_240527_101226/checkpoint/I1000000_E41 -o ./finetune/coco_adamw_4x/I1000000_E41  -c ./config/IXI_64_128.json
# python tool_add_control.py -i ./experiments/4x_coco_adamw_240527_101226/checkpoint/I1000000_E41 -o ./finetune/coco_adamw_4x/I1000000_E41  -c ./config/IXI_64_128.json


##consis
#multi
# python tool_add_control.py -i ./experiments/4x_c_consis_att1_240718_023812/checkpoint/I1000000_E11 -o ./finetune_new/consistency/I1000000_E11  -c ./config/coco_128_256.json
# python tool_add_control.py -i ./experiments/4x_c_att_t_multi_240724_050323/checkpoint/I810000_E9 -o ./finetune_new/multi/I810000_E9  -c ./config/multi_log.json



import argparse
import model as Model
import logging
import core.logger as Logger
import core.metrics as Metrics

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


# model = create_model(config_path='./models/cldm_v15.yaml')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                            help='JSON file for configuration')
    parser.add_argument('-i','--input_path', type=str, help='Path to the input model file.')
    parser.add_argument('-o','--output_path', type=str, help='Path for the output file.')

    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    opt["is_control"]=False 

    diffusion_without = Model.create_model(opt)
    pretrained_weights = diffusion_without.netG.state_dict()


    # pretrained_weights = diffusion_without.netG.load_state_dict(torch.load( args.input_path+ '_gen.pth') )


    ###change to load_state_dict(torch.load( args.input_path+ '_gen.pth') )
    # import pdb;pdb.set_trace()

    opt["is_control"]=True   ##check model.py files


    optimizer = Model.create_model(opt).optG
    diffusion = Model.create_model(opt)
    scratch_dict = diffusion.netG.state_dict()

    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']


    target_dict = {}
    for k in scratch_dict.keys():
       
        is_control, name = get_node_name(k, 'denoise_fn.unet')
        print(is_control, name)
        if is_control:
            copy_k = 'denoise_fn' + name
        else:
            copy_k = k
        # if is_control:
        #     import pdb;pdb.set_trace()
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')


    # print(target_dict.keys())
    
    diffusion.netG.load_state_dict(target_dict, strict=True)



    gen_path = args.output_path + '_gen.pth'
    opt_path = args.output_path + '_opt.pth'

    #gen
    state_dict = diffusion.netG.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, gen_path)
    # opt
    file_name = args.input_path.split("/")[-1]
    epoch = file_name.split("_E")[-1]
    iter_step = file_name.split("_E")[0][1:]

    opt_state = {'epoch': int(epoch), 'iter': int(iter_step),'scheduler': None, 'optimizer': None}
    opt_state['optimizer'] = optimizer.state_dict()
    torch.save(opt_state, opt_path)

    # logger.info(
    #     'Saved model in [{:s}] ...'.format(gen_path))
    # torch.save(diffusion.state_dict(), args.output_path)
    print('Done.')


    ######do the inference test#######
    # import pdb;pdb.set_trace()

    import numpy as np
    
    from PIL import Image
    import torchvision.transforms as transforms
    import torch

    def read_png_to_tensor(file_path):
        image = Image.open(file_path).convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        return image_tensor

    def save_array_as_png(array, file_path):
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        image = Image.fromarray(array)
        image.save(file_path)


    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    diffusion_without.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    import data as Data
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            
 
    for _,  val_data in enumerate(val_loader):
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['SR'])  # uint8


        diffusion_without.feed_data(val_data)
        diffusion_without.test(continous=False)
        visuals = diffusion_without.get_current_visuals()

        hr_img_w = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img_w = Metrics.tensor2img(visuals['LR'])  # uint8
        fake_img_w = Metrics.tensor2img(visuals['SR'])  # uint8  ##????use INF before???

        Metrics.save_img(fake_img_w, "./test_withoutcontrol.png")
        Metrics.save_img(fake_img, "./test_withcontrol.png")
        
        import pdb;pdb.set_trace()
   
    
