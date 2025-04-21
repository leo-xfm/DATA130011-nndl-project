from PIL import Image, ImageEnhance
import random
import numpy as np

def random_rotate(image, angle=None):
    if angle is None:
        angle = random.randint(-8, 8)
    return image.rotate(angle)

def random_translate(image):
    max_translation = 3
    translation = (random.randint(-max_translation, max_translation),
                   random.randint(-max_translation, max_translation))
    return image.transform(image.size, Image.AFFINE, (1, 0, translation[0], 0, 1, translation[1]))

def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.9, 1.1)
    return enhancer.enhance(factor)

def random_zoom(image):
    scale_factor = random.uniform(0.9, 1.1)
    new_size = tuple([int(scale_factor * s) for s in image.size])
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = (resized_image.width - image.width) / 2
    top = (resized_image.height - image.height) / 2
    right = (resized_image.width + image.width) / 2
    bottom = (resized_image.height + image.height) / 2
    return resized_image.crop((left, top, right, bottom))

def transform_image(image):
    image = random_rotate(image)
    image = random_translate(image)
    image = random_brightness(image)
    image = random_zoom(image)
    return np.array(image)