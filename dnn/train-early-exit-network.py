#!/usr/bin/env python3

import os
import sys
import torch
from torch.utils.data import DataLoader
import argparse

rundir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(rundir))

from utils.functions import *
from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--glob', type=str, default='2016_01', help='Glob pattern for dataset - default is 2016_01')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training - default is 1000')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for - default is 5')
parser.add_argument('--dataset-folder', type=str, help='Dataset folder to get the data from', required=True)
parser.add_argument('--model', type=str, default='alexnet', help='Model to train', choices=[ 'alexnet', 'mobilenet' ])
parser.add_argument('--output-folder', type=str, default='saves', help='Output folder for the model - default is "saves"')
args = parser.parse_args()

glob = args.glob
batch_size = args.batch_size
epochs = args.epochs
directory = args.dataset_folder
output_folder = args.output_folder

batch_size = 1000
train_data   = CustomDataset(glob=glob, as_expanded_matrix=True, directory=directory)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
elif args.model == 'mobilenet':
    model = MobileNetV2WithExits().to(device)
else:
    print("Invalid model option. Please choose either 'alexnet' or 'mobilenet'.")
    sys.exit(1)

train_2exits(model, train_loader=train_loader, device=device, epochs=epochs, save_path=output_folder)