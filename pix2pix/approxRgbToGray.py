import numpy as np

# Linear approximation of converting RGB channel image to a grayscale image which is a single channel
# This method is faster as compared to the luminous method which is cost demanding

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])