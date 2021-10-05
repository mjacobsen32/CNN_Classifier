from .pad_to_224 import pad_to_224
import imgaug
import numpy as np


def get_image(image):
  padded_image = pad_to_224(np.array(image,dtype='uint8'))
  seq3 = imgaug.augmenters.Sequential([
    imgaug.augmenters.pillike.FilterSmooth(),
    imgaug.augmenters.pillike.FilterEdgeEnhance(),
    imgaug.augmenters.Sharpen()
  ])
  images_aug3 = seq3(images=[padded_image])
  return(np.asarray(images_aug3[0], dtype='float32'))