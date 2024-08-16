#!/usr/bin/env python3

import torch
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import sys
import os

rundir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rundir)
sys.path.append(os.path.dirname(rundir))

from utils.functions import *
from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from calibration.temperature_scaling_2exits import ModelWithTemperature

os.environ['PYTHONUBUFFERED'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--trained-model',
                    help='.pth file to open',
                    required=True)

parser.add_argument('--model',
                    help='Model to choose - [alexnet | mobilenet]',
                    default='alexnet')

parser.add_argument('--batch-size',
                    help='Batch size',
                    default=1000,
                    type=int)

parser.add_argument('--dataset',
                    help='Dataset to use',
                    required=True)

parser.add_argument('--savefile',
                    help='File to save to',
                    required=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
else:
    model = MobileNetV2WithExits().to(device)

model_t = ModelWithTemperature(model, device=device)
model_t.load_state_dict(torch.load(args.trained_model))
model_t.model.eval()

df = pd.read_csv(args.dataset)

show_accuracy = 1
if len(df.columns) == 58:
    show_accuracy = 0
    df['class'] = 0

data   = CustomDataset(as_matrix=True, df=df)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

total = 0
accuracy_total = [ 0, 0 ]
certainty_total = [ 0, 0 ]
dfs = []
for b, (X, y) in enumerate(loader):
    X = X.to(device)
    y = y.to(device)
    count = len(y)
    total += count

    y_pred = model_t(X) 
    
    df = pd.DataFrame()

    for exit, y_pred_exit in enumerate(y_pred):   
        certainty, predicted = torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)
        
        df[f'exit_{exit + 1}_certainty'] = certainty.cpu().detach().numpy()
        df[f'exit_{exit + 1}_prediction'] = predicted.cpu().detach().numpy()
        
        accuracy_total[exit] += (predicted == y).sum().item()
        certainty_total[exit] += certainty.sum().item()
    
    if show_accuracy:
        df['class'] = y.cpu().numpy()
        
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.to_csv(args.savefile, index=False)

if show_accuracy:
    for exit in range(2):
        accuracy = accuracy_total[exit] / total
        avg_certainty = certainty_total[exit] / total
        print(f'Exit {exit + 1}: Accuracy: {accuracy:.3f}, Avg Certainty: {avg_certainty:.3f}')