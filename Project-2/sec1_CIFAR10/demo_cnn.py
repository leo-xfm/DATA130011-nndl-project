#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import time
from tqdm import tqdm
import numpy as np

from utils.visualize import visualize_dataset
from utils.get_dataset import get_dataset
from models.myModel import myCNNModel

import matplotlib.pyplot as plt

import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7899'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7899'

train_dataset, val_dataset, test_dataset = get_dataset(val_ratio=0.1)
# visualize_dataset()

batch_size = 256
num_epochs = 5000
lr = 0.01
patience = 50

saved_model_name = time.strftime("%Y%m%d-%H%M%S", time.localtime())
saved_model_path = f"saved_models/{saved_model_name}_bestCNN"

print(saved_model_path)

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
    
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = myCNNModel(num_classes=10,
                 cfg=[128, 256, 512, 512],# [64, 128, 256, 512], [32, 64, 128, 256], [128, 256, 512, 512]
                 res=True, 
                 dropout=0.8 # 0, 0.2, 0.4, 0.6, 0.8
                 ) 

from timm.loss import LabelSmoothingCrossEntropy
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# criterion = nn.CrossEntropyLoss()
 
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
# optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, 
    # milestones=[int(num_epochs*p) for p in [0.025, 0.03, 0.08, 0.2, 0.5, 0.8]], 
    milestones=[50, 70, 120, 200, 500, 1000, 2500, 3000], 
    # milestones=[10, 20, 40, 60, 80, 100, 200, 500, 1000, 2500, 3000], 
    # milestones=[50, 100, 200, 500, 1000, 2500, 3000], 
    gamma=0.1, 
    last_epoch = -1 
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("model_parameters:", count_parameters(model))

# Train!!!
def fire(model, loss_fn, optimizer, epochs, saved_model_path, patience):

    model.to(device)
    # summary(model, input_size=(3, 32, 32))
    
    history = []
    
    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0

    loop_bar = tqdm(range(epochs))

    for epoch in loop_bar:
        loop_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            ret, pred = torch.max(outputs.data, 1)
            correct_counts = pred.eq(labels.data.view_as(pred))
            acc = correct_counts.float().mean()
            train_acc += acc.item() * inputs.size(0)
        
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                ret, pred = torch.max(outputs.data, 1)
                correct_counts = pred.eq(labels.data.view_as(pred))
                acc = correct_counts.float().mean()
                val_acc += acc.item() * inputs.size(0)
            
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_acc / len(train_dataset)
 
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_acc / len(val_dataset)
        
        # torch.save(model, f"{saved_model_path}/checkpoint_{epoch}.pt")
        if best_acc < avg_val_acc:
            best_acc = avg_val_acc
            best_epoch = epoch + 1
            torch.save(model, f"{saved_model_path}/best_model.pt")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        epoch_end = time.time()
        ecplise_time = epoch_end - epoch_start
        
        history.append([avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, ecplise_time])
                
        loop_bar.set_postfix(
            train_loss=f'{avg_train_loss:.4f}',
            val_loss=f'{avg_val_loss:.4f}',
            train_acc=f'{avg_train_acc * 100:.4f}%',
            val_acc=f'{avg_val_acc * 100:.4f}%',
            no_improve_count=no_improve_count
        )
        
        if no_improve_count >= patience:
            return history, best_epoch
        
    return history, best_epoch

history, best_epoch = fire(model, criterion, optimizer, num_epochs, saved_model_path, patience)
print(f"Finish Training, Best Epoch: {best_epoch}")
torch.save(history, f"{saved_model_path}/history.pt")

history = np.array(history)

plt.figure(figsize=(10, 10))
plt.plot(history[:best_epoch, 0:2])
plt.legend(['Train Loss', 'Val Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0, best_epoch + 1, step=5))
# plt.yticks(np.arange(0, 1.05, 0.1))
plt.grid()
plt.savefig(f"{saved_model_path}/loss_curve.png")

plt.figure(figsize=(10, 10))
plt.plot(history[:best_epoch, 2:4])
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, best_epoch + 1, step=5))
plt.yticks(np.arange(0, 1.05, 0.05))
plt.grid()
plt.savefig(f"{saved_model_path}/acc_curve.png")


# Test!!!
model_path = f"{saved_model_path}/best_model.pt"
model = torch.load(model_path)
model.to(device)
model.eval()

test_loss = 0.0
test_acc = 0.0
loss_fn = criterion

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        test_loss += loss.item() * inputs.size(0)
        ret, pred = torch.max(outputs.data, 1)
        correct_counts = pred.eq(labels.data.view_as(pred))
        acc = correct_counts.float().mean()
        test_acc += acc.item() * inputs.size(0)
    

avg_test_loss = test_loss / len(test_dataset)
avg_test_acc = test_acc / len(test_dataset)
print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_acc:.4f}")
