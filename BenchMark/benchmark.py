import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.gridspec import GridSpec
from .markov_sequence import MarkovSequenceGenerator
from .markov_dataset import MarkovDataset
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MarkovBenchmark: 
    def __init__(self, 
                 n_symbols_list, 
                 n_states_list, 
                 dataset_size=1000, 
                 sequence_length=100,

                 # Test kwarg
                 proportion_anomalous=0.5,
                 n_anomaly_generator=5, 
                 nbins=20,
                 seed=42, 
                ): 
        self.n_symbols_list = n_symbols_list
        self.n_states_list = n_states_list
        self.dataset_size = dataset_size
        self.sequence_length = sequence_length
        
        self.proportion_anomalous = proportion_anomalous
        self.n_anomaly_generator = n_anomaly_generator

        self.nbins = nbins
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_dataset_library(self):
        dataset_library = []
        for i, (n_symbols, n_states) in enumerate(zip(self.n_symbols_list, self.n_states_list)): 
            dataset = self.generate_dataset(n_symbols, n_states)
            dataset_library.append(dataset)
        self.dataset_library = dataset_library
        return self.dataset_library
        
    def generate_dataset(self, n_symbols, n_states):
        transition_matrices = []
        train_generator = MarkovSequenceGenerator(n_states=n_states, n_symbols=n_symbols, sequence_length=self.sequence_length, rng=self.rng)
        transition_matrices.append(train_generator.transition_matrix)
        train_set = train_generator.generate_all_sequences(self.dataset_size)

        n_good = int((1 - self.proportion_anomalous) * self.dataset_size)
        n_bad = int((self.dataset_size - n_good) / self.n_anomaly_generator)

        good_data = train_generator.generate_all_sequences(n_good)

        bad_data = []
        for _ in range(self.n_anomaly_generator): 
            anomaly_generator = MarkovSequenceGenerator(n_states=n_states, n_symbols=n_symbols, sequence_length=self.sequence_length, rng=self.rng)
            anomaly = anomaly_generator.generate_all_sequences(n_bad)
            transition_matrices.append(anomaly_generator.transition_matrix)
            bad_data.extend(anomaly)
        
        test_set = good_data + bad_data
        labels = [0 for _ in range(n_good)] + [1 for _ in range(n_bad * self.n_anomaly_generator)]
        return MarkovDataset(train_set, test_set, labels, n_symbols, n_states, transition_matrices)    

    def plot_distribution(self, n_symbols, n_states, labels, anomaly_scores, ax):      
        good = anomaly_scores[labels == 0]  # Normal sequences
        bad = anomaly_scores[labels == 1]   # Anomalous sequences

        title = f"Distribution of anomaly scores, anomaly vs baseline \n {n_symbols} symbols in sequence "

        common_range = (min(min(good), min(bad)), max(max(good), max(bad)))
        bins = np.linspace(common_range[0], common_range[1], self.nbins + 1)
        ax.hist(good, bins=bins, label="baseline", alpha=0.5, color='blue')
        ax.hist(bad, bins=bins, label="anomaly", alpha=0.5, color='red')
        ax.set_title(title)
        ax.set_xlabel("anomaly score")
        ax.set_ylabel("n samples")
        ax.legend()

    def plot_AUC_score(self, fpr, tpr, auc_score, ax): 
        ax.plot(fpr, tpr, label=f'ROC Curve ')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC = {auc_score:.3f})')
        ax.legend(loc='lower right')
        ax.grid()

    def benchmark(self, model, metrics, dataset_library=None, figsize=(15, 10)):
        if dataset_library is None: 
            dataset_library = self.generate_dataset_library()
         
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Anomaly Score Distributions for Different transition matrix size of the {model.__class__.__name__}", fontsize=16, fontweight='bold')
        n_plot = len(dataset_library)
        gs = GridSpec(n_plot, 3)

        for i, dataset in enumerate(dataset_library): 
            model.train(dataset.train)
            anomaly_scores = model.predict_proba(dataset.test)

            auc_score = roc_auc_score(dataset.labels, anomaly_scores)
            fpr, tpr, _ = roc_curve(dataset.labels, anomaly_scores)

            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            ax3 = fig.add_subplot(gs[i, 2])

            self.plot_distribution(dataset.n_symbols, dataset.n_states, dataset.labels, anomaly_scores, ax1)
            self.plot_AUC_score(fpr, tpr, auc_score, ax2)
            dataset.plot_2D_reduction(metrics, ax=ax3)
            model.reset()

        fig.tight_layout()
        plt.show()
