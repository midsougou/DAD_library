import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.io import arff
import matplotlib.pyplot as plt

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class SAX:
    def __init__(self, word_size, alphabet_size, mode="global"):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.alphabet = ALPHABET[:alphabet_size]
        self.mode = mode
        self.breakpoints = self._compute_breakpoints(alphabet_size)

        self.global_mean = None
        self.global_std = None
        self.raw_dataset = None
        self.dataset = None

    def read_file(self, filename): 
        data, _ = arff.loadarff(filename)

        self.timeseries = []
        self.labels = []

        all_values = []
        for line in data: 
            serie = list(line)
            self.labels.append(int(serie[-1]))
            serie = np.array(list(serie[:-1]))
            self.timeseries.append(serie)
            all_values.extend(serie)  
        
        all_values = np.array(all_values, dtype=float)
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        if self.global_std == 0:
            self.global_std = 1e-12 
        
        return self.timeseries
                
    def _compute_breakpoints(self, alphabet_size):
        quantiles = [(i / alphabet_size) for i in range(1, alphabet_size)]
        breakpoints = norm.ppf(quantiles)

        self.symbol_mapping = {}
        i = 1
        mids = []
        for point1, point2 in zip(breakpoints[:-1], breakpoints[1:]):
            mids.append((point1 + point2) / 2)
            self.symbol_mapping[self.alphabet[i]] = (point1 + point2) / 2
            i += 1

        self.symbol_mapping[self.alphabet[0]] = mids[0] - (mids[0] - breakpoints[0])
        self.symbol_mapping[self.alphabet[-1]] = mids[-1] + (breakpoints[-1] - mids[-1])
        return breakpoints
    
    def _z_normalize(self, sequence):
        """
        Z-normalize a time series using the global mean and std (mean 0, std 1).
        """
        if self.mode == "global": 
            return (np.array(sequence) - self.global_mean) / self.global_std
        elif self.mode == "local": 
            return (np.array(sequence) - sequence.mean()) / sequence.std()
        else: 
            raise ValueError("Please specify a mode `local` or `global` ")
    
    def _paa(self, sequence):
        """
        Piecewise Aggregate Approximation of a time series into word_size segments.
        """
        n = len(sequence)
        idx = np.arange(0, n+1, self.word_size).astype(int)
        paa_values = []
        for idx1, idx2 in zip(idx[:-1], idx[1:]):
            segment = sequence[idx1:idx2]
            paa_values.append(np.mean(segment))
        paa_values = np.array(paa_values)
        return paa_values
    
    def _discretize(self, paa_values):
        """
        Convert PAA values into symbols using the precomputed breakpoints.
        """
        symbols = []
        for val in paa_values:
            idx = np.sum(val > self.breakpoints)  # count how many breakpoints val exceeds
            symbols.append(self.alphabet[idx])
        return symbols
    
    def transform(self):
        self.discreet_sequences = []
        for timeserie in self.timeseries:
            z_normed = self._z_normalize(timeserie)
            paa_vals = self._paa(z_normed)
            sequence = self._discretize(paa_vals)
            self.discreet_sequences.append(sequence)
    
        self.dataset = pd.DataFrame({"sequence": self.discreet_sequences, 
                                     "timeserie": self.timeseries, 
                                     "label": self.labels})
        return self.dataset
    
    def display_sequence(self, sequence_number, figsize=(10, 4)):
        sequence = self.dataset.iloc[sequence_number, 0] 
        serie = self.dataset.iloc[sequence_number, 1]
        label = self.dataset.iloc[sequence_number, 2]

        value_sequence = [self.symbol_mapping[symbol] for symbol in sequence]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Sequence {sequence_number}, label : {label}")
        ax.plot(serie, label="continuous timeserie")
        ax.plot([self.word_size * i for i in range(len(sequence))], value_sequence, color="red", label="discretized sequence")
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.legend()

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())  # Match the limits of the primary y-axis
        ax2.set_yticks([self.symbol_mapping[letter] for letter in list(self.alphabet)])  # Place ticks at the discretized values
        ax2.set_yticklabels(list(self.alphabet))  # Set the labels to the symbols (letters)
        ax2.set_ylabel("Discretized symbols (letters)")
        plt.show()