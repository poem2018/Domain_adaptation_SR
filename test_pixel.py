from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

######check resolution############
# def check_image_resolutions(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png')):
#             image_path = os.path.join(folder_path, filename)
#             with Image.open(image_path) as img:
#                 if img.size[0]!=1024:
#                     print("!!!")
#                 # print(f"{filename}: {img.size}")
#                 # break

# folder_path =  "../dataset/finetune_8x/sr_128_1024" # Replace with your folder path
# check_image_resolutions(folder_path)




# #############remove borke image###############
# def remove_broken_images(directory):
#     removed_count = 0
#     # Iterate through all files in the directory
#     count=0
#     for file_path in Path(directory).glob('*'):
#         if count%1000==0:
#             print(count) 
#         count+=1
#         if file_path.is_file():
#             try:
#                 # Try to open the image
#                 with Image.open(file_path) as img:
#                     # If successful, check if the image is loaded properly
#                     img.verify()
#             except (IOError, SyntaxError) as e:
#                 # If an error occurs, remove the file
#                 print(f"Removing broken image: {file_path}")
#                 os.remove(file_path)
#                 removed_count += 1
#     return removed_count

# # Specify the directory path
# dir_path = './dataset/Celeba_2x_512_1024_256/sr_256_512'

# # Remove broken images and get the count of removed files
# removed_files = remove_broken_images(dir_path)
# print(f"Total removed files: {removed_files}")




# # ############pixel-level variance##############
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = Image.open(os.path.join(folder, filename))
#         if img is not None:
#             images.append(np.asarray(img))
#     return images

# def calculate_pixel_variance(images):
#     # Stack images along a new dimension
#     stack = np.stack(images, axis=0)
#     # Calculate variance along the stack
#     return np.var(stack, axis=0)

# def visualize_variance(variance):
#     plt.imshow(variance/255.0, cmap='hot')
#     plt.colorbar()
#     plt.title("Pixel-wise Variance")
#     plt.savefig('./variance.png', format='png')

# # Folder containing images
# folder = './vis_variance'

# # Load images
# images = load_images_from_folder(folder)
# variance = calculate_pixel_variance(images)
# visualize_variance(variance)



import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.asarray(img))
    return images

# Function to calculate pixel standard deviation
def calculate_pixel_variance(images):
    # Stack images along a new dimension
    stack = np.stack(images, axis=0)
    # Calculate variance along the stack
    return np.var(stack, axis=0)

# Function to visualize multiple variance maps in one figure
def visualize_multiple_variances(variances, rows, cols, output_file='variance_maps.png', vmin=0, vmax=34):
    fig = plt.figure(figsize=(15, 5 * rows))  # Adjust height by row count

    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[0.1] + [1] * cols, height_ratios=[1] * rows)

    for i, variance in enumerate(variances):
        ax = fig.add_subplot(gs[i // cols, (i % cols) + 1])
        im = ax.imshow(variance / 255.0, cmap='hot', vmin=vmin / 255.0, vmax=vmax / 255.0)
        ax.set_title(f"Variance Map for sample {i + 1}", fontdict={'fontsize': 20,  'family': 'sans-serif'})
        ax.axis('off')

    # Add a colorbar to the left side, making it span all rows and match height with subplots
    cbar_ax = fig.add_subplot(gs[:, 0])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)  # Adjust tick label size if needed

    # Reduce the spacing between subplots
    # plt.subplots_adjust(wspace=0.18, hspace=0.3)  # Reduced wspace to bring subplots closer
    # Reduce spacing between subplots and adjust margins
    plt.subplots_adjust(left=0.01, right=0.98, top=0.95, bottom=0.05, wspace=0.17, hspace=0.1)

    plt.savefig(output_file, format='png')
    plt.show()

# Folder containing images
folders = ['./vis_variance/fig1', './vis_variance/fig2', './vis_variance/fig3']  # List of folders containing images for different variance maps

# Calculate variance for each folder
variances = []
for folder in folders:
    images = load_images_from_folder(folder)
    variance = calculate_pixel_variance(images)
    variances.append(variance)

# Visualize all variance maps in one figure
visualize_multiple_variances(variances, rows=1, cols=len(variances), vmin=0, vmax=34)  # Adjust 'rows' and 'cols' as needed
