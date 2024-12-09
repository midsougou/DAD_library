import numpy as np
from collections import defaultdict

class FixedMarkovianBased:
    def __init__(self, k=2):
        """
        k: The length of history to condition on (k-length subsequences).
        """
        self.k = k
        self.transition_counts = defaultdict(int)  # tracks how often each (k-1)-length context is followed by a specific symbol
        self.context_counts = defaultdict(int)  # tracks how often each context (k-1) appears in the training data
    
    def train(self, sequences):
        """
        Train the model by counting the frequencies of subsequences of length k.
        """
        for sequence in sequences:
            for i in range(len(sequence) - self.k + 1):
                context = tuple(sequence[i:i+self.k-1])  # (k-1)-length context
                symbol = sequence[i + self.k - 1]  # Next symbol
                self.transition_counts[context + (symbol,)] += 1
                self.context_counts[context] += 1

    def compute_conditional_probability(self, context, symbol):
        """
        Compute P(symbol | context) as:
        P(symbol | context) = f(context + symbol) / f(context)
        """
        if self.context_counts[context] == 0:
            return 0.001  # Smoothing to avoid division by zero
        return self.transition_counts[context + (symbol,)] / self.context_counts[context]
    
    def compute_sequence_probability(self, sequence):
        """
        Compute the total probability of a test sequence.
        """
        probability = 1.0
        for i in range(len(sequence) - self.k + 1):
            context = tuple(sequence[i:i+self.k-1])  # (k-1)-length context
            symbol = sequence[i + self.k - 1]  # Next symbol
            probability *= self.compute_conditional_probability(context, symbol)
        return probability

    def compute_anomaly_score(self, sequence):
        """
        Compute the anomaly score for the sequence as the inverse of the probability.
        """
        probability = self.compute_sequence_probability(sequence)
        if probability == 0:
            return float('inf')  # If the probability is zero, anomaly score is infinite
        return 1 / probability
