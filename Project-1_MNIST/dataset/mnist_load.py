import numpy as np
from struct import unpack

def load_images(file_name):
    with open(file_name, 'rb') as f:
        magic, num_imgs, rows, cols = unpack('>4I', f.read(16))
        print("magic, num_imgs, rows, cols:", magic, num_imgs, rows, cols)
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_imgs, rows, cols)
    return imgs

def load_labels(file_name):
    with open(file_name, 'rb') as f:
        magic, num_labels = unpack('>2I', f.read(8))
        print("magic, num_labels:", magic, num_labels)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels