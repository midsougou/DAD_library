import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.manifold import MDS

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MarkovDataset: 
    def __init__(self, train_set, test_set, labels, n_symbols, n_states, transition_matrices):
        self.train = np.array(train_set)
        self.test = np.array(test_set)
        self.labels = np.array(labels)
        self.n_symbols = np.array(n_symbols)
        self.n_states = np.array(n_states)
        self.transition_matrices = transition_matrices

        self.alphabet = list(ALPHABET[:n_symbols])
        self.symbol_mapping = {elt: i for i, elt in enumerate(self.alphabet)}

        self.test_baseline = self.test[self.labels == 0]  # Normal sequences
        self.test_anomaly = self.test[self.labels == 1]   # Anomalous sequences

    def plot_sequence(self, sequence, ax=None):
        if ax == None: 
            fig, ax = plt.subplots()
        
        for sequence in sequence: 
            sequence = [self.symbol_mapping[elt] for elt in sequence]
            ax.plot(sequence)
        
        ax.set_yticks([self.symbol_mapping[elt] for elt in self.alphabet])  # Place ticks at the discretized values
        ax.set_yticklabels(list(self.alphabet))  # Set the labels to the symbols (letters)
        ax.set_ylabel("Discretized symbols (letters)")
        return ax
    
    def plot_dataset(self, n_sequence=2, figsize=(10, 5)): 
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        rows = np.random.choice(self.test_baseline.shape[0], size=n_sequence, replace=False)
        baseline_sample = self.test_baseline[rows, :]
        self.plot_sequence(baseline_sample, ax=ax1)
        ax1.set_title("Baseline Markov chain")

        rows = np.random.choice(self.test_anomaly .shape[0], size=n_sequence, replace=False)
        anomaly_sample = self.test_anomaly[rows, :]
        self.plot_sequence(anomaly_sample, ax=ax2)
        ax2.set_title("Anomalous Markov chain")
        fig.tight_layout()

    def plot_heatmap(self, figsize=(15, 10)):
        baseline = self.transition_matrices[0]
        n = len(self.transition_matrices) 
        n_plot = int((n+1)/ 2)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_plot) 
        axes = [fig.add_subplot(gs[i > n_plot - 1, i % n_plot]) for i in range(n-1)]

        for i, matrix in enumerate(self.transition_matrices[1:]): 
            diff = np.abs(baseline - matrix)
            sns.heatmap(diff, annot=True, fmt=".1f", cmap="coolwarm", cbar=False, ax=axes[i])
            axes[i].set_title(f"Anomaly {i+1} - Baseline")
            axes[i].set_xlabel("Columns")
            axes[i].set_ylabel("Rows")

        fig.suptitle("Heatmaps of Absolute Differences Compared to Baseline", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
        plt.show()

    def compute_distance_matrix(self, metrics): 
        n = len(self.transition_matrices)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = metrics(self.transition_matrices[i], self.transition_matrices[j])
        return distance_matrix

    def plot_2D_reduction(self, metrics, figsize=(8, 8)): 
        distance_matrix = self.compute_distance_matrix(metrics)

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        points_2D = mds.fit_transform(distance_matrix)

        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.scatter(points_2D[:, 0], points_2D[:, 1], color='blue', s=100)
        for i, (x, y) in enumerate(points_2D):
            ax.text(x, y, f"Matrix {i+1}", fontsize=12, ha='center', va='center')

        ax.set_title("2D Representation of Matrices using MDS")
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.grid()
        plt.show()

   


        



