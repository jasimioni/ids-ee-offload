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
from temperature_scaling_2exits import ModelWithTemperature
# from temperature_scaling_2exits_brute_force import ModelWithTemperature # alternative algorithm for temperature scaling

parser = argparse.ArgumentParser()

parser.add_argument('--trained-model',
                    help='.pth file to open',
                    required=True)

parser.add_argument('--calibrated-model-savefile',
                    help='.pth file to save',
                    required=True)

parser.add_argument('--model',
                    help='Model to choose - [alexnet | mobilenet]',
                    default='alexnet')

parser.add_argument('--batch-size',
                    help='Batch size',
                    default=1000,
                    type=int)

parser.add_argument('--max-iter', 
                    help='Max iterations for temperature scaling',
                    default=10,
                    type=int)

parser.add_argument('--epochs',
                    help='Number of epochs for training',
                    default=80,
                    type=int)

parser.add_argument('--dataset',
                    help='Dataset to use',
                    required=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
else:
    model = MobileNetV2WithExits().to(device)

model.load_state_dict(torch.load(args.trained_model))

print(f"Epochs: {args.epochs}, Max Iterations: {args.max_iter}")
model_t = ModelWithTemperature(model, device=device, max_iter=args.max_iter, epochs=args.epochs)

data   = CustomDataset(as_matrix=True, file=args.dataset)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

model_t.set_temperature(loader)

torch.save(model_t.state_dict(), args.calibrated_model_savefile)

print(model_t.temperature)
print(model_t.temperature[0])
print(model_t.temperature[1])

model_t = ModelWithTemperature(model, device=device, max_iter=args.max_iter, epochs=args.epochs)
model_t.load_state_dict(torch.load(args.calibrated_model_savefile))

print(model_t.temperature)
print(model_t.temperature[0])
print(model_t.temperature[1])
