import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import math
from datetime import datetime
import time
import os
import numpy


class CustomDataset(Dataset):
    def __init__(self, as_matrix=True, as_expanded_matrix=False, glob='200701', 
                 directory='/home/ubuntu/datasets/balanced/', file=None, df=None):
        
        if df is not None:
            self.df_labels = df[['class']].copy()
            self.df = df.drop(columns=['class']).copy()
        else:
            if file is not None:
                print(f'Getting file {file}', file=sys.stderr)
                files = [ file ] 
            else:
                print(f'Getting files from {directory}', file=sys.stderr)
                files = Path(directory).glob(f'*{glob}*')

            dfs = []
            for file in sorted(files):
                print("Reading: ", file, file=sys.stderr)
                dfs.append(pd.read_csv(file))

            df = pd.concat(dfs, ignore_index=True)

            self.df_labels = df[['class']].copy()
            self.df = df.drop(columns=['class']).copy()

            del df


        if as_expanded_matrix:
            p_columns = len(self.df.columns)

            columns = self.df.columns

            s_size = 48

            total = s_size ** 2
            for i in range(total - p_columns):
                pos = i % p_columns
                self.df[f'{columns[pos]}{i}'] = self.df[columns[pos]]

            import json
            # print(json.dumps(list(self.df.columns), indent=2))

            self.dataset = torch.tensor(self.df.to_numpy()).float().view(len(self.df), 1, s_size, s_size)

            print(self.dataset[0])
        elif as_matrix:
            p_columns = len(self.df.columns)
            s_size = int(math.sqrt(p_columns)) + 1

            for i in range(s_size**2 - p_columns):
                self.df[f'EmptyCol{i}'] = 0

            self.dataset = torch.tensor(self.df.to_numpy()).float().view(len(self.df), 1, s_size, s_size)
        else:
            self.dataset = torch.tensor(self.df.to_numpy()).float()

        del self.df


        # print(f"Checking: {self.df_labels['class'][0]}")

        if isinstance(self.df_labels['class'][0], str):
            idx = { 'normal' : 0, 'attack' : 1 }
            self.df_labels['class'] = self.df_labels['class'].apply(lambda x: idx[x])
        
        self.labels = torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        del self.df_labels

        # print(self.dataset.shape, file=sys.stderr)
        # print(self.labels.shape, file=sys.stderr)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def train_1exit(model, train_loader=None, lr=0.001, epochs=5, save_path='saves',
                device='cpu', criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):


    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = model.__class__.__name__

    path = os.path.join(save_path, name, dt_string)
    Path(path).mkdir(parents=True, exist_ok=True)
    
    optimizer = optimizer(model.parameters(), lr=lr)

    start_time = time.time()

    seq = 0
    for i in range(epochs):
        trn_cor = 0
        trn_cnt = 0
        tst_cor = 0
        tst_cnt = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            b+=1
            
            y_pred = model(X_train)  
                
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()

            predicted = torch.max(y_pred.data, 1)[1]
            batch_cor = (predicted == y_train).sum()
            trn_cor += batch_cor
            trn_cnt += len(predicted)

            if (b-1)%10 == 0:
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss.item():4.4f} Accuracy Train: {trn_cor.item()*100/trn_cnt:2.3f}%')


            seq += 1

        accuracy = f'{trn_cor.item()*100/trn_cnt:2.3}'

        filename = os.path.join(path, f'epoch_{i}_{accuracy}.pth')
        torch.save(model.state_dict(), filename)
        
        # show_exits_stats(model, test_loader, criterion, device)
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

def train_2exits(model, train_loader=None, lr=0.001, epochs=5, save_path='saves',
                 device='cpu', criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):


    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = model.__class__.__name__

    Path(save_path).mkdir(parents=True, exist_ok=True)   
    
    optimizer = optimizer(model.parameters(), lr=lr)

    start_time = time.time()

    seq = 0
    for i in range(epochs):
        trn_cor = [0, 0]
        cnf = [0, 0]
        trn_cnt = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            b+=1
            
            y_pred = model(X_train)  
                
            losses = [weighting * criterion(res, y_train) for weighting, res in zip(model.exit_loss_weights, y_pred)]

            optimizer.zero_grad()        
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()
            optimizer.step()
            
            for exit, y_pred_exit in enumerate(y_pred):   
                predicted = torch.max(y_pred_exit.data, 1)[1]
                cnf = torch.mean(torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)[0]).item()
                batch_corr = (predicted == y_train).sum()
                trn_cor[exit] += batch_corr
                    
            trn_cnt += len(predicted)
            
            if (b-1)%10 == 0:
                loss_string = [ f'{loss.item():4.4f}' for loss in losses ]
                accu_string = [ f'{correct.item()*100/trn_cnt:2.3}%' for correct in trn_cor ]
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss_string} Accuracy Train: {accu_string}%')

            seq += 1

        accuracy = '_'.join(f'{correct.item()*100/trn_cnt:2.3}' for correct in trn_cor)

        filename = os.path.join(save_path, f'{name}_{dt_string}_epoch_{i}_{accuracy}.pth')
        torch.save(model.state_dict(), filename)
        
        # show_exits_stats(model, test_loader, criterion, device)
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')


def evaluate_2exits(model, device='cpu', loader=None, batch_size=1000):
    accumulated_correct = [0, 0]
    accumulated_count = 0

    for b, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        b += 1

        correct = [ 0, 0 ]
        count = len(X)
        accumulated_count += count
                
        y_pred = model(X)  

        # print(y_pred)
                            
        for exit, y_pred_exit in enumerate(y_pred):   
            predicted = torch.max(y_pred_exit.data, 1)[1]
            correct[exit] = (predicted == y).sum()
            accumulated_correct[exit] += correct[exit]
                        
        accu_string = [ f'{corr.item()*100/count:2.3}%' for corr in correct ]
        print(f'Batch: {b:3} Accuracy: {accu_string}%')

        if b % 10 == 0:
            accu_string = [ f'{corr.item()*100/accumulated_count:2.3}%' for corr in accumulated_correct ]
            print(f'Accumulated:: {b:3} Accuracy Train: {accu_string}%')

    accu_string = [ f'{corr.item()*100/accumulated_count:2.3}%' for corr in accumulated_correct ]
    print(f'Accumulated:: {b:3} Accuracy Train: {accu_string}%')


def dump_2exits(model, device='cpu', loader=None, savefile=None):
    measurement_mode = model.measurement_mode
    model.set_measurement_mode()

    model(torch.rand(1, 1, 8, 8).to(device))

    df = pd.DataFrame()

    for b, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        b += 1

        print(f'{b:05d}: {len(y)}')

        y_pred = model(X) 

        line = y.view(-1, 1).cpu().numpy().tolist()

        for exit, results in enumerate(y_pred):   
            count = len(y)

            y_pred_exit, bb_time, exit_time = results

            certainty, predicted = torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)

            avg_bb_time = bb_time / count
            avg_exit_time = exit_time / count

            for n in range(count):
                line[n].extend([ predicted[n].item(), certainty[n].item(), avg_bb_time, avg_exit_time ])

        line_df = pd.DataFrame(line)#, columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                     #                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ])

        line_df.columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ]

        correct_exit_1 = (line_df['y'] == line_df['y_exit_1']).sum()
        correct_exit_2 = (line_df['y'] == line_df['y_exit_2']).sum()
        total = len(line_df)

        print(f'Accuracy: Exit 1: {100*correct_exit_1/total:.2f} | {100*correct_exit_2/total:.2f}')
        
        df = pd.concat([ df, line_df ], ignore_index=True)
    

    correct_exit_1 = (df['y'] == df['y_exit_1']).sum()
    correct_exit_2 = (df['y'] == df['y_exit_2']).sum()
    total = len(df)

    print(f'Final Accuracy: Exit 1: {100*correct_exit_1/total:.2f} | {100*correct_exit_2/total:.2f}')

    if savefile is not None:
        df.to_csv(savefile, index=False)

    model.set_measurement_mode(measurement_mode)
    
    return df

def dump_1exit(model, device='cpu', directory='/home/ubuntu/datasets/MOORE/', glob='2016_01', batch_size=1000, savefile='mycsv'):
    data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    measurement_mode = model.measurement_mode
    model.set_measurement_mode()

    model(torch.rand(1, 1, 8, 8).to(device))

    df = pd.DataFrame()

    for b, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        b += 1

        print(f'{b:05d}: {len(y)}')

        results = model(X) 

        line = y.view(-1, 1).cpu().numpy().tolist()

        count = len(y)

        y_pred, total_time = results

        certainty, predicted = torch.max(nn.functional.softmax(y_pred, dim=-1), 1)

        print(f"Total Time: {total_time}, Count: {count}")
        avg_time = total_time / count

        for n in range(count):
            line[n].extend([ predicted[n].item(), certainty[n].item(), avg_time ])

        line_df = pd.DataFrame(line)#, columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                     #                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ])
        
        df = pd.concat([ df, line_df ], ignore_index=True)
    
    df.columns = [ 'y', 'y_pred', 'cnf', 'avg_time' ]

    df.to_csv(savefile, index=False)

    model.set_measurement_mode(measurement_mode)
