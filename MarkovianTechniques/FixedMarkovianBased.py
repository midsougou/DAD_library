import numpy as np
from collections import defaultdict
from .markov_struct import MarkovStruct

class FixedMarkovianBased(MarkovStruct):
    def __init__(self, max_depth=2):
        """
        max_depth: The length of history to condition on (max_depth-length subsequences).
        """
        super().__init__()
        self.max_depth = max_depth
        self.transition_counts = defaultdict(int)  # tracks how often each (max_depth-1)-length context is followed by a specific symbol
        self.context_counts = defaultdict(int)  # tracks how often each context (max_depth-1) appears in the training data
    
    def train(self, sequences):
        """
        Train the model by counting the frequencies of subsequences of length max_depth.
        """
        for sequence in sequences:
            for i in range(len(sequence) - self.max_depth + 1):
                context = tuple(sequence[i:i+self.max_depth-1])  # (max_depth-1)-length context
                symbol = sequence[i + self.max_depth - 1]  # Next symbol
                self.transition_counts[context + (symbol,)] += 1
                self.context_counts[context] += 1

        proba = self.predict_proba(sequences)
        self.bound = np.percentile(proba, 95)

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
        for i in range(len(sequence) - self.max_depth + 1):
            context = tuple(sequence[i:i+self.max_depth-1])  # (max_depth-1)-length context
            symbol = sequence[i + self.max_depth - 1]  # Next symbol
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