from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class DataInspector: 
    def __init__(self, dataset): 
        self.dataset = dataset
        self.n_symbols = self.count_symbols(dataset)

    def to_array(self, array):  
        return np.array([elt for elt in array])

    def count_symbols(self, dataset): 
        symbols = set()
        for sequence in dataset["sequence"].values:
            symbols = symbols.union(set(sequence))
        self.symbols = symbols

    def one_hot_encode_array(self, array):
        mapping = {elt: i for i, elt in enumerate(self.symbols)}
        one_hot_encoded = []

        for subarray in array: 
            cat = np.zeros((len(subarray), len(self.symbols)))
            for i, elt in enumerate(subarray): 
                cat[i, mapping[elt]] = 1
            cat = cat.flatten()
            one_hot_encoded.append(cat)
    
        return np.array(one_hot_encoded)

    def continuous_separability(self, C=10):
        series = self.dataset["timeserie"].values
        labels = self.dataset["label"].values - 1
        self.series = np.array([np.array(elt) for elt in series])
        X_train, X_test, y_train, y_test = train_test_split(self.series, labels, random_state=10)

        self.continuous_svm = SVC(C=C, random_state=42)
        self.continuous_svm.fit(X_train, y_train)
        self.continuous_score = self.continuous_svm.score(X_test, y_test)
        return self.continuous_score
    
    def symbolic_separability(self, C=10): 
        sequences = self.dataset["sequence"].values
        labels = self.dataset["label"].values - 1

        self.sequences = self.one_hot_encode_array(sequences)
        X_train, X_test, y_train, y_test = train_test_split(self.sequences, labels, random_state=42)
        self.symbolic_svm = SVC(C=C, random_state=42)
        self.symbolic_svm.fit(X_train, y_train)
        self.symbolic_score = self.symbolic_svm.score(X_test, y_test)
        return self.symbolic_score
    
    def plot_t_SNE(self, data, mode, ax):
        X_train, X_test, y_train, y_test = train_test_split(data, self.labels, test_size=0.3, random_state=42)
        tsne = TSNE(n_components=2, random_state=42)
        X_train_2d = tsne.fit_transform(X_train)
        X_test_2d = tsne.fit_transform(X_test) 

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(X_train_2d, y_train)
        score = svm.score(X_test_2d, y_test)
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor="k", label="Training data")
        ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolor="k", label="Training data")
        ax.set_title(f"{mode} data, score : {score}")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend()

    def display_decision_boundary_t_SNE(self, figsize=(10, 5)): 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        fig.suptitle("Decision Boundary in 2D (t-SNE Projection)")
        self.plot_t_SNE(self.sequences, mode="symbolic", ax=ax1)
        self.plot_t_SNE(self.series, mode="continuous", ax=ax2)
        fig.tight_layout()
        plt.show()

    def pca(self, dataset=None):
        if dataset is None: 
            dataset = self.dataset
         
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

    def display_pca(self, figsize=(10, 10), dataset=None): 
        self.pca(dataset=dataset)
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



    

