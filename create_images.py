import os
from imgaug import augmenters as iaa
import PIL
import os
import png
import numpy as np

def get_perimeter_pixels(image):
    perimeter_pixels = []
    x, y = image.shape
    for (count, i) in enumerate(image):
        for j, j_val in enumerate(i):
            if(count == 0) or (count == x-1) or (j == 0) or (j == y-1):
                perimeter_pixels.append(j_val)
    return(perimeter_pixels)

def get_background(perimeter_pixels):
    i = np.random.choice(a=perimeter_pixels, size=(224,224))
    return(i)

def pad_to_224(image):
    perimeter_pixels = get_perimeter_pixels(image)
    background = get_background(perimeter_pixels)
    background_image = PIL.Image.fromarray(np.uint8(background), 'L')
    image_pil = PIL.Image.fromarray(np.uint8(image), 'L')
    x, y = image_pil.size
    x = int(x/2)
    y = int(y/2)
    offset = (112 - x, 112 - y)
    background_image.paste(image_pil, offset)
    return np.asarray(background_image, dtype='uint8')

def get_image(image, aug):
    augs = iaa.Sequential([])
    if aug == 'aug1': 
        augs = iaa.Sequential([iaa.pillike.FilterFindEdges(),iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug2': 
        augs = iaa.Sequential([iaa.pillike.FilterEdgeEnhanceMore(), iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug3': 
        augs = iaa.Sequential([
            iaa.pillike.FilterSmooth(),
            iaa.pillike.FilterEdgeEnhance(),
            iaa.Sharpen(),
            iaa.Resize({'height': 224, 'width': 224})
            ])
    elif aug == 'aug4': 
        augs = iaa.Sequential([iaa.pillike.Autocontrast(), iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug5': 
        augs = iaa.Sequential([iaa.pillike.FilterEmboss(),iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug6': 
        augs = iaa.Sequential([iaa.pillike.FilterContour(),iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug7': 
        augs = iaa.Sequential([iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'aug8': 
        augs = iaa.Sequential([imgcorruptlike.GaussianNoise(severity=2),iaa.Resize({'height': 224, 'width': 224})])
    elif aug == 'edge_padding':
        width, height = image.shape
        if (width > 224) or (height > 224):
            if width < height:
                augs = iaa.Sequential([iaa.Resize({'height': 'keep-aspect-ratio', 'width': 224}),
                                    iaa.PadToFixedSize(width=224,height=224, position='center', pad_mode='edge')])
            else:
                augs = iaa.Sequential([iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'}),
                                    iaa.PadToFixedSize(width=224,height=224, position='center', pad_mode='edge')])
        else:
            augs = iaa.Sequential([iaa.PadToFixedSize(width=224, height=224, position='center', pad_mode='edge')])
    elif aug == 'black_padding':
        width, height = image.shape
        if (width > 224) or (height > 224):
            if width < height:
                augs = iaa.Sequential([iaa.Resize({'height': 'keep-aspect-ratio', 'width': 224}),
                                    iaa.PadToFixedSize(width=224,height=224, position='center')])
            else:
                augs = iaa.Sequential([iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'}),
                                    iaa.PadToFixedSize(width=224,height=224, position='center')])
        else:
            augs = iaa.Sequential([iaa.PadToFixedSize(width=224, height=224, position='center')])
    elif aug == 'resize':
        augs = iaa.Resize({'height': 224, 'width': 224})
    elif aug == 'resize_with_pad':
        width, height = image.shape
        if width < height:
            augs = iaa.Sequential([iaa.Resize({'height': 'keep-aspect-ratio', 'width': 224}),
                                    iaa.CenterPadToFixedSize(width=224,height=224)])
        else:
            augs = iaa.Sequential([iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'}),
                                    iaa.CenterPadToFixedSize(width=224,height=224)])
    elif aug == 'random_padding':
        width, height = image.shape
        if (width > 224) or (height > 224):
            if width < height:
                augs = iaa.Sequential([iaa.Resize({'height': 'keep-aspect-ratio', 'width': 224})])
            else:
                augs = iaa.Sequential([iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'})])
            image = augs(images=[image])[0]
        return(pad_to_224(image))
    else:
        print("NO AUG SELECTED")
    image = augs(images=[image])
    #return(image[0])
    return(np.asarray(image[0], dtype='uint8'))

def create_images(folder_path, operation, original_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for d in os.listdir(original_path):
        if not os.path.exists(folder_path+'/'+str(d)):
            os.mkdir(folder_path+'/'+str(d))
            imagesList = os.listdir(original_path+str(d))
            print("imageList: {}".format(imagesList))
            for image in imagesList:
                pic = get_image(np.array(PIL.Image.open(original_path + d + '/' + image), dtype=np.uint8), operation)
                png.from_array(pic, 'L').save(folder_path + '/' + d + '/' + image)
            print('dir: {} completed'.format(d))
    print("images completed")
