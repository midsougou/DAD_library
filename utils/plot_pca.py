import numpy as np
from sklearn.decomposition import PCA

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class PlotPCA: 
    def __init__(self):
        pass

    def to_array(self, array):  
        return np.array([elt for elt in array])

    def get_symbols(self, dataset): 
        symbols = set()
        for sequence in dataset["sequence"].values:
            symbols = symbols.union(set(sequence))
        self.symbols = symbols
    
    def one_hot_encode_array(self, array):
        mapping = {elt: i for i, elt in enumerate(self.symbols)}
        print(mapping)
        one_hot_encoded = []

        for subarray in array: 
            cat = np.zeros((len(subarray), len(self.symbols)))
            for i, elt in enumerate(subarray): 
                cat[i, mapping[elt]] = 1
            cat = cat.flatten()
            one_hot_encoded.append(cat)
    
        return np.array(one_hot_encoded)
    
    def pca(self, dataset): 
        self.get_symbols(dataset)
        timeseries = self.to_array(dataset["timeserie"].values)
        sequences = self.one_hot_encode_array(dataset["sequence"].values)

        self.labels = np.array(dataset["label"].values - 1, dtype=bool)
        self.pca_series = PCA()
        self.series_result = self.pca_series.fit_transform(timeseries)
        
        self.pca_sequence = PCA()
        self.sequence_result = self.pca_sequence.fit_transform(sequences)

    def plot_explained_variance(self, mode, ax): 
        if mode == "symbolic": 
            explained_variance_ratio = self.pca_sequence.explained_variance_ratio_
        elif mode == "continuous": 
            explained_variance_ratio = self.pca_series.explained_variance_ratio_

        ax.set_title(f"{mode} ")
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
        ax.grid()

    def plot_projection(self, mode, ax):
        if mode == "symbolic": 
            projected_data = self.sequence_result[:, :2]
        elif mode == "continuous": 
            projected_data = self.series_result[:, :2]

        ax.scatter(projected_data[self.labels, 0], projected_data[self.labels, 1], c='blue', alpha=0.7, label="class 1")
        ax.scatter(projected_data[~self.labels, 0], projected_data[~self.labels, 1], c='orange', alpha=0.7, label="class 2")
        ax.set_title(f"Projection of {mode} data along principal vectors")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid()

    def display(self, figsize=(10, 10)): 
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self.plot_explained_variance(mode="symbolic", ax=ax1)
        self.plot_explained_variance(mode="continuous", ax=ax2)

        self.plot_projection(mode="symbolic", ax=ax3)
        self.plot_projection(mode="continuous", ax=ax4)
        fig.tight_layout()
        plt.show()