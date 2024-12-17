from .markov_sequence import MarkovSequenceGenerator
from .markov_dataset import MarkovDataset
import random

class MarkovDatasetGenerator: 
    def __init__(self, transition_matrices, emission_matrices, n_sequences=100, sequence_length=50): 
        self.transition_matrices = transition_matrices
        self.emission_matrices = emission_matrices 
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.generators = self.init_transform()
    
    def init_transform(self):
        self.generators = []
        self.dataset = []
        for transition_matrix, emission_matrix in zip(self.transition_matrices, self.emission_matrices): 
            generator = MarkovSequenceGenerator(transition_matrix=transition_matrix, 
                                           emission_matrix=emission_matrix, 
                                           n_sequences=self.n_sequences, 
                                           sequence_length=self.sequence_length) 
            self.generators.append(generator)
        return self.generators

    def generate(self): 
        self.dataset = []
        self.labels = []
        for i, generator in enumerate(self.generators): 
            sequences = generator.generate_all_sequences()
            self.labels.extend([i for _ in range(len(sequences))])
            self.dataset.extend(sequences)

        permutation = list(range(len(self.labels)))
        random.shuffle(permutation)

        shuffled_dataset = [self.dataset[i] for i in permutation]
        shuffled_labels = [self.labels[i] for i in permutation]

        self.dataset = shuffled_dataset
        self.labels = shuffled_labels
        return self.dataset, self.labels