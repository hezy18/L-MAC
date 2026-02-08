import pandas as pd
import numpy as np
import math
from config.config import DATA_PATH, FEATURES, DATA_ROOT, DATA_SETTING, DATA_DATE_VERSION, USE_IMAGE
from utils.metrics import evaluate_predictions
from utils.data_loader_writer import save_multiple_matrices_to_csv
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.sparse import issparse

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FeatureSelection(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureSelection, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, feed_dict, flat_emb):
        # flat_emb: [B, D]
        weights = self.gate(flat_emb)        # [B, D]
        feat1 = flat_emb * weights           
        feat2 = flat_emb * (1 - weights)     
        return feat1, feat2


# ========== InteractionAggregation ==========
class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(
            num_heads * self.head_x_dim * self.head_y_dim, output_dim
        ))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        batch_size = x.shape[0]
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(batch_size, self.num_heads, self.head_x_dim).flatten(0, 1)
        head_y = y.view(batch_size, self.num_heads, self.head_y_dim).flatten(0, 1)

        xy = torch.matmul(
            torch.matmul(
                head_x.unsqueeze(1),
                self.w_xy.view(self.num_heads, self.head_x_dim, -1)
            ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.unsqueeze(-1)
        ).squeeze(-1)

        xy_reshape = xy.sum(dim=1).view(batch_size, -1)
        output += xy_reshape
        return output.squeeze(-1)


# ========== FinalMLP ==========
class FinalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2, use_fs=True, num_classes=1):
        super(FinalMLP, self).__init__()
        self.use_fs = use_fs
        self.num_classes = num_classes
        if use_fs:
            self.fs_module = FeatureSelection(input_dim, hidden_dim=64)

        def build_mlp(dims):
            layers, last_dim = [], input_dim
            for h in dims:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                last_dim = h
            return nn.Sequential(*layers)

        self.mlp1 = build_mlp(hidden_dims)
        self.mlp2 = build_mlp(hidden_dims)

        output_dim = 1 if num_classes == 1 else num_classes 
        self.aggregation = InteractionAggregation(hidden_dims[-1], hidden_dims[-1], output_dim=output_dim, num_heads=1)

    def forward(self, x):
        if self.use_fs:
            feat1, feat2 = self.fs_module(x, x)
        else:
            feat1, feat2 = x, x

        out1 = self.mlp1(feat1)
        out2 = self.mlp2(feat2)

        out = self.aggregation(out1, out2)   # [B, 1]
        if self.num_classes == 1:
            out = torch.sigmoid(out).squeeze(-1)
        else:
            out = out.squeeze(-1) if out.dim() == 2 else out  # [B, num_classes]
        return out

def to_tensor(X, y=None, device="cpu"):
    if issparse(X):
        X = X.toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        return X_tensor, y_tensor
    return X_tensor

def run_FinalMLP(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                 model_names: str = 'all', label_type: str = 'reg', attr_setting: str = 'all', 
                 hidden_dims=[256, 128], dropout=0.2, lr=1e-3, batch_size=64, epochs=5):
    if label_type in ['select','filter']:
        num_classes = 2
    else:
        num_classes = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = to_tensor(X_train_top, y_train_df.values.ravel(), device)
    X_dev, y_dev = to_tensor(X_dev_top, y_dev_df.values.ravel(), device)
    X_test, y_test = to_tensor(X_test_top, y_test_df.values.ravel(), device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = FinalMLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout = dropout, use_fs=True, num_classes=num_classes).to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss() if num_classes==1 else nn.CrossEntropyLoss()

    best_dev_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            # loss = criterion(preds, yb)
            # import pdb; pdb.set_trace()
            if num_classes == 1:
                loss = criterion(preds, yb.float())
            elif num_classes == 2:
                loss = criterion(preds, yb.long().squeeze())
            else:
                loss = criterion(preds, (yb.long()-1).squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            dev_preds = model(X_dev)
            if num_classes == 1:
                dev_loss = criterion(dev_preds, y_dev.float())
            elif num_classes == 2:
                dev_loss = criterion(dev_preds, y_dev.long().squeeze())
            else:
                dev_loss = criterion(dev_preds, (y_dev.long()-1).squeeze())
        # if dev_loss < best_dev_loss:
        if True:
            best_dev_loss = dev_loss
            best_state = model.state_dict()
        print(total_loss/len(train_loader), dev_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}, Dev Loss: {dev_loss:.4f}")
    # import pdb; pdb.set_trace()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if num_classes == 1:
            y_test_pred = (model(X_test).cpu().numpy() > 0.5).astype(int)
            y_dev_pred = (model(X_dev).cpu().numpy() > 0.5).astype(int)
        else:
            y_test_pred = (torch.argmax(model(X_test).cpu(), dim=1) + 1).numpy().astype(int)
            y_dev_pred = (torch.argmax(model(X_dev).cpu(), dim=1) + 1).numpy().astype(int)

    # import pdb; pdb.set_trace()
    
    if num_classes == 2:
        dev_metrics, dev_cm = evaluate_predictions(y_dev.cpu().numpy().astype(int), y_dev_pred, num_class=2)
        metrics, cm = evaluate_predictions(y_test.cpu().numpy().astype(int), y_test_pred, num_class=2)
    else:
        dev_metrics, dev_cm = evaluate_predictions(y_dev.cpu().numpy().astype(int), y_dev_pred)
        metrics, cm = evaluate_predictions(y_test.cpu().numpy().astype(int), y_test_pred)
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    # import pdb; pdb.set_trace()
    return metrics, cm, dev_metrics, dev_cm, y_test_pred


def run_FinalMLP_gridsearch(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                 model_names: str = 'all', label_type: str = 'all', attr_setting: str = 'all'):
    results = {}
    # cms = {} 
    preds = {}      
    for hidden_dims in [[256, 128], [128, 64], [512, 256]]:
        for dropout in [0,0.1, 0.2,0.3,0.4]:
            for lr in [1e-3, 1e-4, 1e-5]: #
                metrics, cm, dev_metrics, dev_cm, y_test_pred = run_FinalMLP(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                            model_names, label_type, attr_setting, 
                            hidden_dims, dropout, lr)
                results[f'{label_type}_{hidden_dims}_{dropout}_{lr}'] = {}
                results[f'{label_type}_{hidden_dims}_{dropout}_{lr}']['dev']= dev_metrics
                results[f'{label_type}_{hidden_dims}_{dropout}_{lr}']['test'] = metrics
                preds[f'{label_type}_{hidden_dims}_{dropout}_{lr}'] = y_test_pred.tolist()
    

    with open('FinalMLP_ali_tmp_select.json', 'w') as f:
        json.dump(preds, f, indent=4)

    output_path = DATA_PATH['baseline_metrics_path']
    
    with open(f'{output_path.split('.')[0]}_{attr_setting}_FinalMLP_{label_type}.{output_path.split('.')[1]}', 'w') as f:
        json.dump(results, f, indent=4)

    # save_multiple_matrices_to_csv(list(cms.values()), list(cms.keys()), f'{output_path.split('.')[0]}_cm_{attr_setting}_FinalMLP.csv')