import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from camvid import CamVid
from model import BayesCenterSegNet
from utils import PILToLongTensor, pixel_accuracy, to_numpy
from torch.utils.data import DataLoader
from torchvision import transforms

def train_step(model, dataloader, optimizer, criterion, device):
    logs = []
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_logit = model(X)
        acc = pixel_accuracy(y_logit, y)
        loss = criterion(torch.log(y_logit), y)
        loss.backward()
        optimizer.step()
        logs.append([to_numpy(loss), to_numpy(acc)])
    return np.array(logs)

def evaluate(model, dataloader, device, k=1, use_dropout=True, reduce_mean=True):
    if use_dropout:
        model.train()
    else:
        model.eval()
    
    with torch.no_grad():
        accuracy = 0
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # monte carlo samples
            y_logits = []
            for _ in range(k):
                y_logits.append(model(X))
            y_logits = torch.stack(y_logits, dim=0).squeeze(0)
            if reduce_mean and k > 1:
                y_logits = y_logits.mean(0)
            acc = pixel_accuracy(y_logits, y)
            accuracy += acc
            num_batch = i
            
        accuracy /= num_batch
        
    return np.array([to_numpy(acc)])

def main():
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 500
    batch_size = 5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_root = "/mnt/e/data/CamVid/SegNet-Tutorial/CamVid"
    
    transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((360, 480)),
        PILToLongTensor()
    ])

    train_data = CamVid(data_root, "train", transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = CamVid(data_root, "val", transform=transform, target_transform=target_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    class_weights = torch.tensor(
        [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 
         9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0.0], 
        device=device)
    
    model = BayesCenterSegNet(in_channels=3, out_channels=12)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss(weight=class_weights, ignore_index=11)
    
    train_logs, val_logs = [], []
    for i in range(epochs):
        train_log = train_step(model, train_loader, optimizer, criterion, device)
        print("Epoch {}, last mini-batch nll={}, acc={}".format(i+1, train_log[-1][0], train_log[-1][1]))
        train_logs.append(train_log)
        
        val_log = evaluate(model, val_loader, device, 10, True, True)
        print("Epoch {}, val acc={}".format(i+1, val_log[0]))
        val_logs.append(val_log)
    
    train_logs, val_logs = np.concatenate(train_logs, axis=0), np.concatenate(val_logs, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.arange(train_logs.shape[0]), train_logs[:, 0], 'r-', label='nll')
    ax2.plot(np.arange(train_logs.shape[0]), train_logs[:, 1], 'r-', label='acc')
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel("batch")
    plt.savefig("train_bayessegnet")
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    ax1.plot(np.arange(val_logs.shape[0]), val_logs[:, 0], 'r-', label='acc')
    for ax in [ax1]:
        ax.legend()
        ax.set_xlabel("epoch")
    plt.savefig("test_bayessegnet")
    
if __name__ == "__main__":
    main()
