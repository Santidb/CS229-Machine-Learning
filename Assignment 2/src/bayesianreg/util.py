import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt

# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """
    df = pd.read_csv(csv_path, converters={'x': from_np_array})
    inputs = np.asarray([a for a in df['x'].to_numpy()])
    labels = df['y'].to_numpy()

    return inputs, labels

def plot(val_err, save_path, n_list):
    """Plot dataset size vs. val err for different reg strengths

    Args:
        val_err: Matrix of validation erros, row.
        save_path: Path to save the plot.
        n_list: List of trainset sizes.
    """
    # Plot dataset
    plt.figure()
    for i in range(len(scale_list)):
        plt.plot(n_list, val_err[i], linewidth=2, label='lambda=%0.4flamda_opt'%scale_list[i])

    # Add labels and save to disk
    plt.xlabel('Num Samples')
    plt.ylabel('Validation Err')
    plt.ylim(0,2)
    plt.legend()
    plt.savefig(save_path)
