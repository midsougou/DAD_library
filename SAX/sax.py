import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.io import arff

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.io import arff

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class SAX:
    def __init__(self, word_size, alphabet_size, mode="global"):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.breakpoints = self._compute_breakpoints(alphabet_size)
        self.alphabet = ALPHABET[:alphabet_size]

        self.mode = mode

        self.global_mean = None
        self.global_std = None
        self.raw_dataset = None
        self.dataset = None

    def read_file(self, filename): 
        data, _ = arff.loadarff(filename)

        dataset = []
        all_values = []
        for line in data: 
            serie = list(line)
            target = serie[-1]
            serie = np.array(list(serie[:-1]))
            dataset.append((serie, target))
            all_values.extend(serie)  
        
        all_values = np.array(all_values, dtype=float)
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        if self.global_std == 0:
            self.global_std = 1e-12  # Avoid division by zero
        
        self.raw_dataset = dataset
        return dataset
                
    def _compute_breakpoints(self, alphabet_size):
        quantiles = [(i / alphabet_size) for i in range(1, alphabet_size)]
        breakpoints = norm.ppf(quantiles)
        return breakpoints
    
    def _z_normalize(self, sequence):
        """
        Z-normalize a time series using the global mean and std (mean 0, std 1).
        """
        if self.mode == "global": 
            return (np.array(sequence) - self.global_mean) / self.global_std
        else: 
            return (np.array(sequence) - sequence.mean()) / sequence.std
    
    def _paa(self, sequence, word_size):
        """
        Piecewise Aggregate Approximation of a time series into word_size segments.
        """
        n = len(sequence)
        if n % word_size == 0:
            frames = np.split(sequence, word_size)
            paa_values = np.array([frame.mean() for frame in frames])
        else:
            idx = np.linspace(0, n, word_size+1, endpoint=True).astype(int)
            paa_values = []
            for i in range(word_size):
                segment = sequence[idx[i]:idx[i+1]]
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
        sax_sequences = []
        for sequence, label in self.raw_dataset:
            z_normed = self._z_normalize(sequence)
            paa_vals = self._paa(z_normed, self.word_size)
            symbols = self._discretize(paa_vals)
            sax_sequences.append({"sequence":symbols, "label": label})
        
        self.dataset = pd.DataFrame(sax_sequences)
        return self.dataset