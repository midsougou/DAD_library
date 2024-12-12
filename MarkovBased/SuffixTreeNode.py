from collections import defaultdict

class SuffixTreeNode:
    """A node in the probabilistic suffix tree."""
    def __init__(self):
        self.children = {}  # Children of the node (representing next states)
        self.counts = defaultdict(int)  # Count of next symbols that follow this context
        self.total_count = 0  # Total number of times this context was observed

class ProbabilisticSuffixTree:
    """A suffix tree to store variable-length contexts."""
    def __init__(self, max_depth=3):
        self.root = SuffixTreeNode()
        self.max_depth = max_depth

    def insert(self, sequence):
        """Insert all suffixes of the sequence into the tree."""
        for i in range(len(sequence)):
            node = self.root
            for j in range(i, max(i - self.max_depth, -1), -1):
                symbol = sequence[j]
                if symbol not in node.children:
                    node.children[symbol] = SuffixTreeNode()
                node = node.children[symbol]
                node.total_count += 1
                if i < len(sequence) - 1:
                    next_symbol = sequence[i + 1]
                    node.counts[next_symbol] += 1

    def get_probability(self, context, symbol):
        """Backoff logic: try largest context, then smaller contexts."""
        node = self.root
        for state in reversed(context):
            if state in node.children:
                node = node.children[state]
                if symbol in node.counts:
                    return node.counts[symbol] / node.total_count
            else:
                break
        return 0.001  # Smoothing for unseen context-symbol pairs
