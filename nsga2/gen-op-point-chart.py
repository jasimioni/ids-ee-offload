#!/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def remove_dominated(results):
    to_delete = []
    for i in range(len(results)):
        for f in range(len(results)):
            if f == i or ( len(to_delete) and to_delete[-1] == i ):
                continue
            if results[f][1] <= results[i][1] and results[f][2] <= results[i][2]:
                to_delete.append(i)

    # print(f'to_delete: {to_delete}')
    to_delete.reverse()
    for i in to_delete:
        results.pop(i)
    return len(to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="Path to the file to load", required=True)
    parser.add_argument("--savefile", help="Path to the file to save", required=True)
    parser.add_argument("--operation-point", help="Operation point coordinates (Array position of options - def: 5", default=5, type=int)

    args = parser.parse_args()
    
    with open(args.datafile, "rb") as f:
        X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

    results = []
    
    print(f"Min time: {min_time:.2f}us | Max time: {max_time:.2f}us")

    for i in range(len(F)):
        f = F[i]
        x = X[i]
        quality = 100 * (1 - sum(f) / len(f))

        acc  = 100 * ( 1 - f[0] )
        time = min_time + f[1] * (max_time - min_time) 

        string = f'Score: {quality:.2f}% | Acc: {acc:.2f}% Time: {time:.2f}us | {x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}, {x[3]:.4f}'
        results.append([ quality, *f, *x, string ])

        # print(f'{i:02d}: {quality:.2f} {100 * (1-f[0]):.2f} {100*(1-f[1]):.2f} => {string}')

    while remove_dominated(results):
        pass
    
    df = pd.DataFrame(results, columns = ['Quality', 'Accuracy', 'Time', 'n_1', 'a_1', 'n_2', 'a_2', 'String'])
    df['Score'] = ( df['Accuracy'] + df['Time'] ) / 2
    df['Distance'] = df['Accuracy'] ** 2 + df['Time'] ** 2

    df = df.sort_values(by='Time')

    seq = 0
    for i, row in df.iterrows():
        print(f"{seq:02d}: {row['Quality']:.2f} {100 * (1-row['Accuracy']):.2f} {100*(1-row['Time']):.2f} => {row['String']}")
        seq += 1  
    
    plt.rc('font', family='Times New Roman', size=36)

    y_max = 11
    y_min = 6
    
    x_max = 30
    x_min = -0.5

    colors = [ 'black', 'red', 'blues' ]
    markers = [ 'o', '^' ]

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 6.5))
    ax.set(ylim=(y_min, y_max), ylabel='Error Rate (%)')
    arr_x = 14
    arr_y = 8.3

    y = 1 - df['Accuracy'].to_numpy()
    x = df['Time'].to_numpy()
    score = df['Score'].to_numpy()
    names = df['String'].to_numpy()    

    y1 = 100 * ( 1 - y )
    x1 = x * 100

    operation_point = df.iloc[args.operation_point]
    print(operation_point)
    y_op = operation_point['Accuracy']
    y1_op = 100 * y_op
    x_op = operation_point['Time']
    x1_op = x_op * 100

    ax.arrow(arr_x, arr_y, x1_op - arr_x, y1_op - arr_y, head_width=0.2, head_length=0.8, 
             fc='k', ec='k', length_includes_head=True, zorder=15)

    ax.plot(x1, y1, linewidth=6)

    avg_time = min_time + x_op * ( max_time + min_time )
    print(f'Error rate: {y1_op}% | Time: {1000 * avg_time:.2f}ms')
    
    ax.text(arr_x - 3, arr_y + 0.2, 'Operation Point')
    ax.legend(loc='upper right', frameon=False)
    # Não é o ganho em percentual, mas sim um valor calculado do ganho, onde
    # ficar igual ao mínimo = 0 e ficar igual ao máximo = 1
    ax.set(xlim=(x_min, x_max), xlabel='Norm. Energy Consump. (%)')
    fig.savefig(args.savefile)    
