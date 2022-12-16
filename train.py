import torch
import argparse
import numpy as np
from torch import nn
from camvid import CamVid
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torchvision import transforms
from model import BayesCenterSegNet
from torch.utils.data import DataLoader
from utils import PILToLongTensor, to_numpy
from torchmetrics import Accuracy, JaccardIndex

def train_step(model, dataloader, optimizer, criterion, device):
    logs = []
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_logit = model(X)
        pred = torch.argmax(y_logit, 1)
        acc = pred.eq(y.data.view_as(pred)).float().cpu().mean()
        loss = criterion(torch.log(y_logit), y)
        loss.backward()
        optimizer.step()
        logs.append([to_numpy(loss), to_numpy(acc)])
    return np.array(logs)

def evaluate(model, dataloader, metrics, device, use_dropout=True):
    k = 10
    
    if use_dropout:
        model.train()
    else:
        model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # monte carlo samples
            y_logit = []
            for _ in range(k):
                y_logit.append(model(X))
            y_logit = torch.stack(y_logit, dim=0).squeeze(0)
            y_logit = y_logit.mean(0)
            
            for metric in metrics:
                metric.update(y_logit, y)
        
    return np.array([to_numpy(metric.compute()) for metric in metrics])

def main(args):
    lr = 5e-4
    weight_decay = 5e-4
    epochs = 500
    batch_size = 5
    device = args.device
    data_root = args.data_path
    
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
    test_data = CamVid(data_root, "test", transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    class_weights = torch.tensor(
        [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 
         9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0.0], 
        device=device)
    
    model = BayesCenterSegNet(in_channels=3, out_channels=12)
    model.to(device)
    print(model)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss(weight=class_weights, ignore_index=11)
    metrics = [
        Accuracy(task="multiclass", num_classes=12, ignore_index=11).to(device),
        JaccardIndex(task="multiclass", num_classes=12, ignore_index=11).to(device)
    ]
    
    train_logs, val_logs = [], []
    for i in range(epochs):
        train_log = train_step(model, train_loader, optimizer, criterion, device)
        print("Epoch {}, last mini-batch nll={}, acc={}".format(i+1, train_log[-1][0], train_log[-1][1]))
        train_logs.append(train_log)
        
        val_log = evaluate(model, val_loader, metrics, device, True)
        print("Epoch {}, val acc={}, iou={}".format(i+1, val_log[0], val_log[1]))
        val_logs.append(val_log[np.newaxis])
        
        for metric in metrics:
            metric.reset()
    
    train_logs, val_logs = np.concatenate(train_logs, axis=0), np.concatenate(val_logs, axis=0)
    
    test_log = evaluate(model, test_loader, metrics, device, True)
    print("Epoch {}, test acc={}, iou={}".format(i+1, test_log[0], test_log[1]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.arange(train_logs.shape[0]), train_logs[:, 0], 'r-', label='nll')
    ax2.plot(np.arange(train_logs.shape[0]), train_logs[:, 1], 'r-', label='acc')
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel("batch")
    plt.savefig("train_bayessegnet")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.arange(val_logs.shape[0]), val_logs[:, 0], 'r-', label='acc')
    ax2.plot(np.arange(val_logs.shape[0]), val_logs[:, 1], 'r-', label='iou')
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel("epoch")
    plt.savefig("val_bayessegnet")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data2/users/yn621/SegNet-Tutorial/CamVid", help="data directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    args = parser.parse_args()
    
    main(args)
