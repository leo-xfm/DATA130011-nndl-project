import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mynn as nn
from dataset.mnist_load import *

############################################ Settings ############################################
np.random.seed(309)

model = nn.Model(module = nn.module.MLP([784, 600, 10], 'Sigmoid', [1e-4, 1e-4]),
                optimizer = nn.optimizer.SGD(init_lr=0.06),
                scheduler = nn.lr_scheduler.MultiStepLR(milestones=[800, 2400, 4000], gamma=0.5),
                loss_fn = 'CrossEntropyLoss'
                )

save_folder = "../saved_models/idx07" #!!!!!

epochs = 10000
bsz = 4096
patience = 500 

val_size = 10000
##################################################################################################

train_images = load_images('../dataset/train-images-idx3-ubyte').reshape(-1, 784) / 255.0
train_labels = load_labels('../dataset/train-labels-idx1-ubyte')
test_images = load_images('../dataset/t10k-images-idx3-ubyte').reshape(-1, 784) / 255.0
test_labels = load_labels('../dataset/t10k-labels-idx1-ubyte')
indices = np.random.permutation(len(train_images))
train_indices = indices[val_size:]
val_indices = indices[:val_size]
train_images, val_images = train_images[train_indices], train_images[val_indices]
train_labels, val_labels = train_labels[train_indices], train_labels[val_indices]

print(f"Training data shape: {train_images.shape}")
print(f"Validation data shape: {val_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation labels shape: {val_labels.shape}")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = f"{save_folder}/best_model.pickle"

X = train_images
y = train_labels
val_X = val_images
val_y = val_labels

train_loss = []
train_score = []
eval_score_ls = []

counter = 0
best_score = 0.0
best_epoch = None
    
for epoch in range(epochs):
    
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]    
    
    model.train()
    running_loss = 0.0
    running_score = 0.0
    
    for it in range(int(X.shape[0] / bsz) + 1):
        
        train_X = X[it * bsz : (it+1) * bsz]
        train_y = y[it * bsz : (it+1) * bsz]

        model.zero_grad()
        output = model(train_X)
        
        loss = model.loss_fn(output, train_y)
        running_loss += loss
        
        model.backward(output, train_y)
        model.optimizerStep()
        model.schedulerStep()
    
    train_loss.append(running_loss)
    
    model.eval()
    score = nn.metric.accuracy(model(X), y)
    train_score.append(score)
        
    y_pred = model(val_X)
    eval_score = nn.metric.accuracy(y_pred, val_y)  
    eval_score_ls.append(eval_score) 
    
    if best_score < eval_score:
        model.save_model(save_path)
        best_epoch = epoch
        best_score = eval_score
        counter = 0
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.5f}, Accuracy: {score * 100:.2f}%, Eval Accuracy: {eval_score * 100:.2f}%, Update best model")
    else:
        counter += 1
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.5f}, Accuracy: {score * 100:.2f}%, Eval Accuracy: {eval_score * 100:.2f}%, patience {counter}")
    
    if counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
        break
    
print(f"Final update of best model: Epoch {best_epoch}, Accuracy {best_score}.")

model.load_model(save_path)
model.eval()
y_pred = model(test_images)
test_score = nn.metric.accuracy(y_pred, test_labels)
print(f"Test Accuracy: {test_score * 100:.2f}%")


with open(f"{save_folder}/training_data.pkl", 'wb') as f:
    pickle.dump({'train_loss': train_loss, 'train_score': train_score, 'eval_score_ls': eval_score_ls, 'test_score': test_score}, f)
    
# with open(f"{save_folder}/training_data.pkl", 'rb') as f:
#     data = pickle.load(f)
# train_loss = data['train_loss']
# train_score = data['train_score']
# eval_score_ls = data['eval_score_ls']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss', color='blue', marker='o', markersize=0.5)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_score, label='Train Accuracy', color='green', marker='o', markersize=0.5)
plt.plot(eval_score_ls, label='Val Accuracy', color='red', marker='o', markersize=0.5)
plt.title('Train and Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f"{save_folder}/plot.png")
