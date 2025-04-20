from utils.SignLanguageDataset import SignLanguageDataset
import os
from configs import Config
import torch
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
from train_lstm import create_label_map, GraphConvolution, GCNClassifier
from utils.mediapipe_edges import build_mediapipe_edge_list, build_adjacency_matrix


def calculate_num_classes(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)
    num_classes = len(set(labels))
    return num_classes

def test(model, dataloader, criterion):
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



if __name__ == "__main__":
    config_file = '../code/I3D/configfiles/asl2000.ini'
    configs = Config(config_file)
    
    LABEL_FILE ="../start_kit/WLASL2000.json"
    NSLT_FILE = "../code/I3D/preprocess/nslt_2000.json"
    label_map = create_label_map(label_file=LABEL_FILE, nslt_file=NSLT_FILE)
    label_map_inv = {v:k for k,v in label_map.items()}
    num_classes = len(label_map)


    
    train_data_dirs = ["./data/landmarks/train"]
    test_data_dirs = ["./data/landmarks/test"]
    train_ds = SignLanguageDataset(data_dir=train_data_dirs, data_json=NSLT_FILE, seq_len=100, num_files=3000, label_map=label_map)
    test_ds = SignLanguageDataset(data_dir=test_data_dirs, data_json=NSLT_FILE, seq_len=100, num_files=1255, label_map=label_map)
    train_dl = DataLoader(train_ds, batch_size=configs.batch_size)
    test_dl = DataLoader(test_ds, batch_size=configs.batch_size)

    train_labels = set([label_map_inv[y.item()] for _, y in train_ds])
    test_labels = set([label_map_inv[y.item()] for _, y in test_ds])

    overlap_labels = train_labels & test_labels
    print(f"Train labels: {len(train_labels)}, Test labels: {len(test_labels)}, Overlap: {len(overlap_labels)}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    edge_list = build_mediapipe_edge_list()
    adj_matrix = build_adjacency_matrix(num_nodes=75, edges=edge_list)
    model = GCNClassifier(num_nodes=75, input_dim=2, hidden_dim=128, num_classes=num_classes, adj_matrix=adj_matrix)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()


    state_dict = torch.load("./metrics/train_sign_model_6862_199.pth")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = test(model, train_dl, criterion)
    print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
    test_loss, test_acc = test(model, test_dl, criterion)
    print(f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

