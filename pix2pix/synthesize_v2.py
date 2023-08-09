#!/usr/bin/env python
# coding: utf-8

# import required libraries
from PIL import Image
import numpy as np
import cv2 as cv
import os

# function to synthesize colour images to sketch type
def rgb2sketch(rgb, filter_kernel=7):
    """
    The range of pixel values is often 0 to 255. We divide by 255 to get a range of 0 to 1.
    """
    rgb = np.asarray(rgb) / 255.0
    
    """
    Make pixel values perceptually linear.
    """
    rgb_lin = ((rgb + 0.055) / 1.055) ** 2.4
    i_low = np.where(rgb <= .04045)
    rgb_lin[i_low] = rgb[i_low] / 12.92
    
    """
    Convert *linear* RGB values to *linear* grayscale values.
    """
    red = rgb_lin[:, :, 0]
    green = rgb_lin[:, :, 1]
    blue = rgb_lin[:, :, 2]
    gray_img = (0.2126 * red + 0.7152 * green + 0.0722 * blue)
    
    """
    Make pixel values display-ready.
    """
    gray = 1.055 * gray_img ** (1 / 2.4) - 0.055
    i_low = np.where(gray_img <= .0031308)
    gray[i_low] = 12.92 * gray_img[i_low]
    gray = (gray * 255).astype(np.uint8)
    
    """
    Invert the grayscale image then blur it
    Note: Here we are using bilateralFilter from cv2 and the second argument set as 5 is a suggestion for real time application
    """
    inv_gray = 255-gray
    x = int((inv_gray.shape[1]/255)*(filter_kernel*10))
    y = int((inv_gray.shape[0]/255)*(filter_kernel*10))
    blur_inv_gray = cv.bilateralFilter(inv_gray, filter_kernel, x, y)

    """
    Dodging to create the final sketch image
    """
    front = blur_inv_gray.astype(int) # Very important for multiplication in the next steps
    back = gray.astype(int)
    
    num, denom = 255*front, 255-back
    final_sketch = num/denom
    final_sketch[final_sketch > 255] = 255
    final_sketch[back == 255] = 255

    return final_sketch.astype('uint8')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
images_path = os.path.abspath('../images')
dataset_path = os.path.abspath('../dataset')

for i in os.listdir(images_path):
    img = Image.open(fr"{images_path}/{i}")
    i = i.split('.')[0]
    sketch = rgb2sketch(img)
    sketch = cv.cvtColor(sketch, cv.COLOR_GRAY2RGB)
    merged = np.concatenate((np.asarray(img), sketch), axis=1)
    cv.imwrite(fr"{dataset_path}/{i}.png", cv.cvtColor(merged, cv.COLOR_RGB2BGR))