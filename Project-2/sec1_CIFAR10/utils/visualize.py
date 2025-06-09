import pickle
import matplotlib.pyplot as plt

def plot_images(data, labels, img_names, labels_type, num_images=40):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0) # (C,H,W) to (H,W,C)
        plt.subplot(5, 8, i + 1)
        plt.imshow(img)
        plt.title(f'img_{i}:{labels_type[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dataset(filename_dataset = './data/cifar-10-batches-py/test_batch',
                      filename_obj = './data/cifar-10-batches-py/batches.meta'): 
    
    with open(filename_obj, 'rb') as f:
        obj = pickle.load(f)
        # print(type(obj))
        # print(obj)
        labels_type = obj['label_names']
    
    with open(filename_dataset, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        # print(type(dataset))
        # print(dataset.keys())
        data = dataset[b'data']
        labels = dataset[b'labels']
        img_names = dataset[b'filenames']
    
    plot_images(data, labels, img_names, labels_type)