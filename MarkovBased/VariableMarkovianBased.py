import numpy as np
from MarkovianTechniques.SuffixTreeNode import ProbabilisticSuffixTree
from .markov_struct import MarkovStruct

class VariableMarkovianBased(MarkovStruct):
    """An anomaly detector using variable Markov techniques."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.pst = ProbabilisticSuffixTree(max_depth=max_depth)

    def _train(self, sequences):
        """Train by inserting sequences into the suffix tree."""
        for sequence in sequences:
            self.pst.insert(sequence)

    def compute_sequence_probability(self, sequence):
        """Compute the total probability of the sequence using the PST."""
        log_probability = 0.0
        for i in range(1, len(sequence)):
            context = sequence[max(0, i - self.pst.max_depth):i]
            symbol = sequence[i]
            prob = self.pst.get_probability(context, symbol)
            log_probability += np.log(prob)
        return np.exp(log_probability)

    def compute_anomaly_score(self, sequence):
        """Compute the anomaly score as the inverse of the sequence probability."""
        log_probability = 0.0
        for i in range(1, len(sequence)):
            context = sequence[max(0, i - self.pst.max_depth):i]
            symbol = sequence[i]
            prob = self.pst.get_probability(context, symbol)
            log_probability += np.log(prob)
        return -log_probability  # Use -log(P) as the anomaly score
