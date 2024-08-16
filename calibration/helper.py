#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


plt.rc('font', family='Times New Roman', size=36)

def getBins(y, y_pred, cnf, n_bins=10, lower=0.5, upper=1):
    ece = 0
    step = ( upper - lower ) / n_bins
    bins = []
    for bin in range(n_bins):
        bin_min = lower + bin * step
        bin_max = lower + ( bin + 1 ) * step

        mask = (cnf > bin_min) & (cnf <= bin_max)
        bin_acc = (y[mask] == y_pred[mask]).sum() / mask.sum()
        bin_cnf = cnf[mask].mean()
        
        bins.append({
            'min': bin_min,
            'max': bin_max,
            'acc': bin_acc,
            'cnf': bin_cnf,
            'prop': mask.sum() / len(y)
        })

        ece += np.abs(bin_acc - bin_cnf) * mask.sum()

    ece /= len(y)

    data = { 'ece': ece, 'bins': bins }
    # print(data)
    return data

def genHistogram(y, y_pred, cnf, n_bins=10, lower=0.5, upper=1, title="", filename=""):
    data = getBins(y, y_pred, cnf, n_bins, lower, upper)
    
    # bin_labels = [ f"{bin['min']:.2f}-{bin['max']:.2f} ({100 * bin['prop']:.2f}%)" for bin in data['bins'] ]
    # bin_labels = [ f"{bin['prop']:.2f}" for bin in data['bins'] ]
    bin_labels = [ 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975 ]
    bin_acc = [ 100 * bin['acc'] for bin in data['bins'] ]
    bin_missing = [ max(0, bin['cnf'] - bin['acc']) for bin in data['bins'] ]
    bin_overflow = [ min(0, bin['cnf'] - bin['acc']) for bin in data['bins'] ]
    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 7))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_xlim(0.5, 1)
    
    print(f"Accuracy: {list(bin_acc)}")
    print(f"Missing bin: {bin_missing}")
    
    width = 0.048

    ax.bar(bin_labels, bin_acc, label="Bin Acc.", width=width, color='#AAAAAA', edgecolor='black', alpha=0.8)
    # ax.bar(bin_labels, bin_missing, label="Under Acc.", bottom=bin_acc, color='yellow', hatch='/', width=width, alpha=0.8)
    # ax.bar(bin_labels, bin_overflow, bottom=bin_acc, color='white', hatch='\\', width=width, alpha=0.5, edgecolor='black')
    

    lr = Ridge()

    np_bins = np.array(range(n_bins)).reshape(-1, 1)

    print(np_bins)
    print(bin_acc)

    lr.fit(np_bins, bin_acc)
    # plt.plot(bin_labels, lr.coef_*np_bins+lr.intercept_, color='orange')
    plt.plot(bin_labels, 100 * np.array( [0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975 ]), color='orange')
    # ax.legend(loc="upper left", frameon=False, fontsize=32)
    
    ax.set_xticks([ 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0 ])

    ax.tick_params(axis='x', rotation=90)
    if title != "":
        ax.set_title(f'{title} | ECE: {data["ece"]:.5f}')
        
    box_text = f"ECE: ({data['ece']:.3f})"
    
    ax.text(0.53, 0.92, box_text)
    
    #text_box = AnchoredText(box_text, frameon=False, loc=0, pad=0.5)
    #plt.setp(text_box.patch, facecolor='white', alpha=0.9)
    #plt.gca().add_artist(text_box)
    
    # Add ECE to the footer of the chart
    ax.text(0.28, .90, f"ECE: {data['ece']:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if filename:
        fig.savefig(filename)
    else:
        plt.show()

def calculateECE(y, y_pred, cnf, n_bins=10, min=0.5, max=1):
    return getBins(y, y_pred, cnf, n_bins, min, max)['ece']