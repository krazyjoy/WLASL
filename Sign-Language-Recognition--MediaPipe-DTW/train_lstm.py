from utils.SignLanguageDataset import SignLanguageDataset
import os
from configs import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import json
from collections import Counter
import matplotlib.pyplot as plt
from utils.mediapipe_edges import build_mediapipe_edge_list, build_adjacency_matrix


def create_label_map(label_file: str, nslt_file: str):
    

    with open(label_file, "r") as f:
        label_data = json.load(f)

    with open(nslt_file, "r") as f:
        nslt_data = json.load(f)

    class_set = set()
    label_map = {}
    num_classes = 0
    for entry in label_data:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_id = inst['video_id']
            class_id = nslt_data[video_id]['action'][0]
            label_map[gloss] = class_id

    print("len(label_map): ", len(label_map))
    return label_map


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (B*T, N, F), adj: (N, N)
        out = torch.matmul(adj, x)  # aggregate from neighbors
        out = self.linear(out)      # linear transformation
        return out


class GCNClassifier(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_classes, adj_matrix):
        super(GCNClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)

        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # for temporal dim
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x = x.view(B * T, N, F)        # Merge batch and time

        x = self.relu(self.gc1(x, self.adj))  # (B*T, N, H)
        x = self.relu(self.gc2(x, self.adj))  # (B*T, N, H)

        x = x.view(B, T, N, -1)        # (B, T, N, H)
        x = x.mean(dim=2)             # Mean over nodes → (B, T, H)
        x = x.permute(0, 2, 1)        # (B, H, T) for pooling
        x = self.global_pool(x).squeeze(-1)  # (B, H)

        return self.classifier(x)     # (B, num_classes)


def train(model, dataloader, optimizer, criterion, scheduler):
    
    model.train()
    total_loss, total_correct = 0, 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        # print("output: ", output.shape)
        loss = criterion(output, batch_y)
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (output.argmax(dim=1) == batch_y).sum().item()
    
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)



def evaluate(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)

            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (output.argmax(dim=1) == batch_y).sum().item()

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def calculate_class_weights(dataset, device, num_classes):
    
    labels = [dataset[i][1].item() for i in range(len(dataset))]

    label_counts = Counter(labels)

    # num_classes = len(set(labels))
    freq = [label_counts.get(i, 1) for i in range(num_classes)]

    class_weights = torch.tensor([1 / f for f in freq], dtype=torch.float32).to(device)

    return class_weights


if __name__ == "__main__":
    config_file = '../code/I3D/configfiles/asl2000.ini'
    configs = Config(config_file)

    LABEL_FILE ="../start_kit/WLASL2000.json"
    NSLT_FILE = "../code/I3D/preprocess/nslt_2000.json"
    label_map = create_label_map(label_file=LABEL_FILE, nslt_file=NSLT_FILE)
    num_classes = len(label_map)
    train_data_dirs = ["./data/landmarks/train"]
   
    
    dataset = SignLanguageDataset(data_dir=train_data_dirs, data_json=NSLT_FILE, seq_len=100, num_files=6500, label_map=label_map)
    print(f"train len(dataset): {len(dataset)}, shape(dataset): {dataset[0][0].shape}") # (seq_len frames, 225 size)
    x, y = dataset[0]

    train_len = int(0.7*len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=configs.batch_size)

    #### test #####
    test_data_dirs = ["./data/landmarks/test"]
    test_ds = SignLanguageDataset(data_dir=test_data_dirs, data_json=NSLT_FILE, seq_len=100, num_files=1255, label_map=label_map)
    test_dl = DataLoader(test_ds, batch_size=configs.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = calculate_class_weights(dataset, device, num_classes)

    edge_list = build_mediapipe_edge_list()
    adj_matrix = build_adjacency_matrix(num_nodes=75, edges=edge_list)
    model = GCNClassifier(num_nodes=75, input_dim=2, hidden_dim=128, num_classes=num_classes, adj_matrix=adj_matrix)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    EPOCHS = 300
    best_val_loss = float("inf")
   
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_dl),
        epochs=300,
        pct_start=0.05,
        anneal_strategy='linear'
    )
    print("start training...")
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []


    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_dl, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, val_dl, criterion)
        test_loss, test_acc = evaluate(model, test_dl, criterion)

        train_losses.append(round(train_loss, 4))
        train_accuracies.append(round(train_acc, 4))
        val_losses.append(round(val_loss, 4))
        val_accuracies.append(round(val_acc, 4))
        test_losses.append(round(test_loss, 4))
        test_accuracies.append(round(test_acc, 4))
        
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"best_val_loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), f"./metrics/train_sign_model_{len(dataset)}_{epoch}.pth")
        # else:
        #     trigger += 1
        #     if trigger > patience:
        #         print("Early stopping triggered.")
        #         break
        print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        import pandas as pd
        epochs_range = list(range(1, epoch+2))
        df = pd.DataFrame({
            "Epoch": epochs_range,
            "Train Loss": train_losses,
            "Train Accuracy": train_accuracies,
            "Val Loss": val_losses,
            "Val Accuracy": val_accuracies
        })
        df = df.reset_index(drop=True)
        df.to_csv("./metrics/conv1d_training_log.csv", index=False)


        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss over Epochs', fontsize=18)
        plt.legend(fontsize=14)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Accuracy over Epochs', fontsize=16)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig("./metrics/conv1d.png")

    print("Model saved as sign_model.pth")



