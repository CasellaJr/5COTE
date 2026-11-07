import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sktime.transformations.panel.rocket import MiniRocketMultivariate

from pathlib import Path

#I want a function for plotting the time series data for each client, showing the distribution of features across clients. This will help visualize how the data is split in a vertical federated learning setting.
def plot_vertical_split(X_clients, Y_clients, ds_name):
    n_clients = len(X_clients)
    # each client may have a different number of feature dimensions
    n_timestamps = X_clients[0].shape[2]

    fig, axes = plt.subplots(n_clients, 1, figsize=(10, 5 * n_clients))

    for i in range(n_clients):
        ax = axes[i] if n_clients > 1 else axes
        n_dims_i = X_clients[i].shape[1]
        for dim in range(n_dims_i):
            ax.plot(range(n_timestamps), X_clients[i][:, dim, :].mean(axis=0), label=f'Dim {dim+1}')
        ax.set_title(f'Client {i+1} Feature Distribution')
        ax.set_xlabel('Timestamps')
        ax.set_ylabel('Mean Value')
        ax.legend()

    plt.tight_layout()
    plt.show()
    result_path = Path("plots")
    result_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(result_path / f'vertical_split_{ds_name}.pdf')  # Save the plot as a PDF file