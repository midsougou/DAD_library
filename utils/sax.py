import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.io import arff
import matplotlib.pyplot as plt
from .var import ALPHABET

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

        # Memory init
        self.all_values = []
        self.timeseries = []
        self.labels = []

    def reset_series(self):
        """After ingesting a file, if the purpose is read only, call this method"""
        self.all_values = []
        self.timeseries = []
        self.labels = []

    def interpolate_nans(self, array):
        nans = np.isnan(array)
        if np.any(nans):
            x = np.arange(array.size)
            array[nans] = np.interp(x[nans], x[~nans], array[~nans])
        return list(array) 

    def set_sax_global_variable(self): 
        all_values = np.array(self.all_values)
        all_values = all_values[~np.isnan(all_values)]
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        if self.global_std == 0:
            self.global_std = 1e-12 
    
    def read_file(self, file_path):
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['value', 'is_anomaly'])
        df_clean = df[df['value'] != -1].reset_index(drop=True)
        self.dataset = df_clean
        return self.dataset
    def ingest_pickle(self, filename): 
        df = pd.read_pickle(filename)

        series = df["timeseries"].values
        labels = df["labels"].values
        for serie, label in zip(series, labels): 
            serie = self.interpolate_nans(serie)
            self.timeseries.append(serie)
            self.labels.append(label)
            self.all_values.extend(serie)
        
        return self.timeseries, self.labels

    def ingest_arff(self, filename): 
        data, _ = arff.loadarff(filename)

        for line in data: 
            serie = list(line)
            self.labels.append(int(serie[-1]))
            serie = np.array(list(serie[:-1]))
            self.timeseries.append(serie)
            self.all_values.extend(serie) 
        
        return self.timeseries, self.labels
                
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
            return (np.array(sequence) - np.array(sequence).mean()) / np.array(sequence).std()
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
        return np.array(symbols)
    
    def transform_with_sliding_windows(self, window_size=200, stride=100):
        self.set_sax_global_variable()
        self.timeseries = []
        self.discreet_sequences = []
        self.labels = []

        for i in range(0, len(self.dataset) - window_size + 1, stride):
            sub_sequence = self.dataset.iloc[i:i + window_size]
            value_sequence = sub_sequence['value'].values  # Extract the continuous values
            z_normed = self._z_normalize(value_sequence)
            paa_vals = self._paa(z_normed)
            sax_sequence = self._discretize(paa_vals)
            anomaly_flag = int(sub_sequence['is_anomaly'].max() == 1)
            
            self.timeseries.append(value_sequence.tolist())
            self.discreet_sequences.append(list(sax_sequence))
            self.labels.append(anomaly_flag)
        self.discreet_sequences = np.array(self.discreet_sequences)
        self.dataset = pd.DataFrame({
            'sequence': list(self.discreet_sequences),
            'timeserie': self.timeseries,
            'label': self.labels
        })
        return self.dataset
    def transform(self):
        self.set_sax_global_variable()

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
    
    def display_sequence(self, sequence_number, figsize=(10, 4), display_moving_average=False):
        sequence = self.dataset.iloc[sequence_number, 0] 
        serie = self.dataset.iloc[sequence_number, 1]
        label = self.dataset.iloc[sequence_number, 2]

        index = np.arange(len(sequence) * self.word_size)
        moving_average = np.convolve(serie, np.ones(self.word_size) / self.word_size, mode='valid')
        value_sequence = np.array([[self.symbol_mapping[symbol]] * self.word_size for symbol in sequence])
        value_sequence = value_sequence.flatten()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Sequence {sequence_number}, label : {label}")
        ax.plot(serie, label="continuous timeserie")
        if display_moving_average:
            ax.plot(moving_average, color="black", label="moving average")
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")

        ax2 = ax.twinx()
        ax2.plot(index, value_sequence, color="red", label="SAX sequence")
        ax2.set_yticks([self.symbol_mapping[letter] for letter in list(self.alphabet)])  # Place ticks at the discretized values
        ax2.set_yticklabels(list(self.alphabet))  # Set the labels to the symbols (letters)
        ax2.set_ylabel("Discretized symbols (letters)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.show()