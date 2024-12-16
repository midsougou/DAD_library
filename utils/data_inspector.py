from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class DataInspector: 
    def __init__(self, dataset): 
        self.dataset = dataset
        
        self.n_symbols = self.count_symbols(dataset)

        series = self.dataset["timeserie"].values
        self.series = np.array([np.array(elt) for elt in series])
        sequences = self.dataset["sequence"].values
        self.sequences = self.one_hot_encode_array(sequences)
        self.labels = dataset["label"].values - 1

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
    
    def train_test_svm(self, data, labels, C=10):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=10)
        svm = SVC(C=C, random_state=42)
        svm.fit(X_train, y_train)
        decision_score =  svm.decision_function(X_test)
        auc_score = roc_auc_score(y_test, decision_score,)
        fpr, tpr, thresholds = roc_curve(y_test, decision_score)
        return auc_score, fpr, tpr

    def continuous_separability(self, C=10, ax=None):
        auc_score, fpr, tpr = self.train_test_svm(self.series, self.labels, C=C)
        return auc_score
    
    def symbolic_separability(self, C=10, ax=None): 
        auc_score, fpr, tpr = self.train_test_svm(self.sequences, self.labels, C=C)            
        return auc_score
    
    def plot_AUC_score(self, auc_score, fpr, tpr, mode, ax): 
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {mode} SVM')
        ax.legend(loc='lower right')
        ax.grid()
        
    def display_AUC_score(self, C=10, figsize=(10, 5)): 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        auc_score, fpr, tpr = self.train_test_svm(self.sequences, self.labels, C=C) 
        self.plot_AUC_score(auc_score, fpr, tpr, "symbolic", ax1)
        auc_score, fpr, tpr = self.train_test_svm(self.series, self.labels, C=C) 
        self.plot_AUC_score(auc_score, fpr, tpr, "continous", ax2)
    
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
        decision_score = svm.decision_function(X_test_2d)
        score = roc_auc_score(y_test, decision_score,)
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor="k", label="Training data")
        ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolor="k", label="Training data")
        ax.set_title(f"{mode} data, auc score : {score}")
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



    

