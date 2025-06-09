import os
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rates = [1e-3, 2e-3, 5e-4, 8e-4]
epochs = 250
batch_size = 2048
save_dir = "saves"
os.makedirs(save_dir, exist_ok=True)

def set_random_seeds(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_npy(arr, prefix, metric, lr):
    filename = f"{prefix}_{metric}_lr_{lr:.0e}.npy"
    path = os.path.join(save_dir, filename)
    np.save(path, np.array(arr, dtype=object), allow_pickle=True)
    print(f"[Saved] {path}")

def compute_max_grad_diff_over_distance(grads_list):
    """
    grads_list: list of epochs; each epoch is list of batch-gradient arrays shape (out_dim, in_dim)
    returns max_{i<j} ||g_i - g_j|| / |i-j|, where g_k is flattened concatenation of all batch grads in epoch k
    """
    # flatten each epoch's grads into 1D
    flat_list = []
    for epoch_grads in grads_list:
        # epoch_grads is list of arrays
        flat = np.concatenate([g.flatten() for g in epoch_grads])
        flat_list.append(flat)
    max_ratio = 0.0
    T = len(flat_list)
    for i in range(T):
        for j in range(i+1, T):
            diff = np.linalg.norm(flat_list[i] - flat_list[j])
            ratio = diff / (j - i)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio

train_loader = get_cifar_loader(train=True,  batch_size=batch_size)
val_loader = get_cifar_loader(train=False, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, train_loader, val_loader, epochs_n):
    model.to(device)
    losses_list = []
    grads_list  = []
    val_accuracy = []

    for _ in tqdm(range(epochs_n), desc="Epochs"):
        model.train()
        epoch_losses = []
        epoch_grads  = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if hasattr(model, 'classifier') and isinstance(model.classifier[-1], nn.Linear):
                epoch_grads.append(model.classifier[-1].weight.grad.detach().cpu().numpy())

        losses_list.append(epoch_losses)
        grads_list.append(epoch_grads)

        # validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv).argmax(dim=1)
                correct += (pred == yv).sum().item()
                total += yv.size(0)
        val_accuracy.append(100 * correct / total)

    return losses_list, grads_list, val_accuracy

for lr in learning_rates:
    # VGG-A
    set_random_seeds(2020)
    print(f"\n[Training VGG_A] lr={lr:.0e}")
    model = VGG_A().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    losses, grads, acc = train(model, optim, train_loader, val_loader, epochs)
    save_npy(losses, "vgg", "loss", lr)
    save_npy(grads, "vgg", "grads", lr)
    save_npy(acc, "vgg", "acc", lr)
    max_grad_dist = compute_max_grad_diff_over_distance(grads)
    np.save(os.path.join(save_dir, f"vgg_maxgraddist_lr_{lr:.0e}.npy"),
            np.array([max_grad_dist]))
    print(f"[Saved] vgg_maxgraddist_lr_{lr:.0e}.npy = {max_grad_dist:.4f}")

    # VGG-A+BN
    set_random_seeds(2020)
    print(f"\n[Training VGG_A+BN] lr={lr:.0e}")
    model_bn = VGG_A_BatchNorm().to(device)
    optim_bn  = torch.optim.Adam(model_bn.parameters(), lr=lr)
    losses, grads, acc = train(model_bn, optim_bn, train_loader, val_loader, epochs)
    save_npy(losses, "bn", "loss", lr)
    save_npy(grads, "bn", "grads", lr)
    save_npy(acc, "bn", "acc", lr)
    max_grad_dist_bn = compute_max_grad_diff_over_distance(grads)
    np.save(os.path.join(save_dir, f"bn_maxgraddist_lr_{lr:.0e}.npy"),
            np.array([max_grad_dist_bn]))
    print(f"[Saved] bn_maxgraddist_lr_{lr:.0e}.npy = {max_grad_dist_bn:.4f}")
