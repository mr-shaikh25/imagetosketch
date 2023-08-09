# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import os
import cv2
from PIL import Image

# load an image
def load_image(filename, size=(256,256)):
    o_size = Image.open(filename).size
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels, o_size

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load source image
src_image, og_size = load_image('trial.png')
print('Loaded', src_image.shape)
# load model
model = load_model("./model_48000.h5")
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
# pyplot.imshow(gen_image[0])
pyplot.imsave("output.png", gen_image[0])
final = cv2.resize(cv2.imread("output.png"), (480,640), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite("output.png", final)