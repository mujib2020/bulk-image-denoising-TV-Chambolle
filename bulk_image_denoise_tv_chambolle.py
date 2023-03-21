# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:57:46 2023

@author: Mujib Chowdhury, PhD

This python script pull images one by one from one folder, then denoise it using denoise_tv_chambolle, 
then save /write them in another directory in png
"""

import os
# from PIL import Image
from skimage import io, color, restoration
# import imageio

input_folder = 'input_folder/images'
output_folder = 'output_folder/denoised'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through the files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        input_image_path = os.path.join(input_folder, filename)
        img = io.imread(input_image_path)
       

        # Convert the image to grayscale if necessary
        if img.ndim == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img
        io.imshow(img)
        
        # Denoise the image using denoise_tv_chambolle
        denoised_img = restoration.denoise_tv_chambolle(img_gray, weight=0.1)
        io.imshow(img)

        # Save the denoised image in the output folder
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_denoised.png")
        io.imsave(output_image_path, (denoised_img * 255).astype('uint8'))

print("Denoising complete!")
