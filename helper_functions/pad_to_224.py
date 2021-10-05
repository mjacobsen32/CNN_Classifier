from .get_background import get_background
from .get_perimeter_pixels import get_perimeter_pixels
import PIL
import numpy as np


def pad_to_224(image):
    perimeter_pixels = get_perimeter_pixels(image)
    background = get_background(perimeter_pixels)
    background_image = PIL.Image.fromarray(np.uint8(background), 'L')
    image_pil = PIL.Image.fromarray(np.uint8(image), 'L')
    x, y = image_pil.size
    x = int(x/2)
    y = int(y/2)
    offset = (112 - x, 112 - y)
    return(background_image.paste(image_pil, offset))
