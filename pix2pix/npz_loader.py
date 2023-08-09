# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from numpy import savez_compressed
from datetime import datetime
import os

# load all images in a directory into memory
def load_images(path, size=(256,512)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(f"{path}/{filename}", target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into rgb and sketch
        rgb_img, sketch_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(rgb_img)
        tar_list.append(sketch_img)
    return [asarray(src_list), asarray(tar_list)]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
npz_path = os.path.abspath('../npz')
# dataset path
dataset_path = os.path.abspath('../dataset')

# load dataset
[src_images, tar_images] = load_images(dataset_path)
print('Loaded: ', src_images.shape, tar_images.shape)

stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# save as compressed numpy array
filename = fr"{npz_path}/sketched_256_{stamp}.npz"
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)