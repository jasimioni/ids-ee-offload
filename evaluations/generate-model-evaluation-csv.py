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

data   = CustomDataset(as_matrix=True, file=args.dataset)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

df = pd.DataFrame()

model_t.model.eval()
model_t.model.set_measurement_mode(True)


for b, (X, y) in enumerate(loader):
    X = X.to(device)
    y = y.to(device)
    b += 1

    print(f'{b:05d}: {len(y)}', end="")

    y_pred = model_t(X) 

    line = y.view(-1, 1).cpu().numpy().tolist()

    for exit, results in enumerate(y_pred):   
        count = len(y)

        y_pred_exit = results[0]

        certainty, predicted = torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)
        accuracy = (predicted == y).sum().item() / len(y)
        avg_certainty = certainty.mean().item()

        print(f' | Exit {exit + 1}: Accuracy: {accuracy:.3f}, Avg Certainty: {avg_certainty:.3f}' , end="")

        avg_bb_time = results[1] / count
        avg_exit_time = results[2] / count

        for n in range(count):
            line[n].extend([ predicted[n].item(), certainty[n].item(), avg_bb_time, avg_exit_time ])

    print("")
    line_df = pd.DataFrame(line)#, columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                 #                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ])
    
    df = pd.concat([ df, line_df ], ignore_index=True)

df.columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                    'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ]

df.to_csv(args.savefile, index=False)
