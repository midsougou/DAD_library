import numpy as np
from collections import defaultdict
from .markov_struct import MarkovStruct

class SparseSuffixTreeNode:
    """A node in the sparse probabilistic suffix tree (with wildcards)."""
    def __init__(self):
        self.children = {}  # Children nodes
        self.counts = defaultdict(int)  # Counts of next symbols
        self.total_count = 0  # Total counts observed

class SparseMarkovTransducer(MarkovStruct):
    """A sparse suffix tree that allows wildcards in contexts."""
    def __init__(self, max_depth=3, wildcard_positions=None):
        self.root = SparseSuffixTreeNode()
        self.max_depth = max_depth
        self.wildcard_positions = wildcard_positions or []  # Positions where wildcards are allowed

    def insert(self, sequence):
        """Insert all sparse suffixes of the sequence."""
        for i in range(len(sequence)):
            node = self.root
            for j in range(i, max(i - self.max_depth, -1), -1):
                position = len(sequence) - j - 1
                symbol = sequence[j] if position not in self.wildcard_positions else '*'
                if symbol not in node.children:
                    node.children[symbol] = SparseSuffixTreeNode()
                node = node.children[symbol]
                node.total_count += 1
                if i < len(sequence) - 1:
                    next_symbol = sequence[i + 1]
                    node.counts[next_symbol] += 1

    def get_probability(self, context, symbol):
        """Get the probability of `symbol` given the `context`."""
        node = self.root
        for position, state in enumerate(reversed(context)):
            state = state if position not in self.wildcard_positions else '*'
            if state in node.children:
                node = node.children[state]
                if symbol in node.counts:
                    return node.counts[symbol] / node.total_count
            else:
                break
        return 0.001  # Default probability if no context is found

    def compute_sequence_probability(self, sequence):
        """Compute the total probability of the sequence using the SMT."""
        log_probability = 0.0
        for i in range(1, len(sequence)):
            context = sequence[max(0, i - self.max_depth):i]
            symbol = sequence[i]
            prob = self.get_probability(context, symbol)
            log_probability += np.log(prob)
        return np.exp(log_probability)

    def compute_anomaly_score(self, sequence):
        """Compute the anomaly score as the inverse of the sequence probability."""
        log_probability = 0.0
        for i in range(1, len(sequence)):
            context = sequence[max(0, i - self.max_depth):i]
            symbol = sequence[i]
            prob = self.get_probability(context, symbol)
            log_probability += np.log(prob)
        return -log_probability  # Use -log(P) as the anomaly score
