import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MarkovSequenceGenerator: 
    def __init__(self, transition_matrix=None, emission_matrix=None, n_states=None, n_symbols=None, sequence_length=50, rng=None): 
        self.transition_matrix = transition_matrix
        self.sequence_length = sequence_length
        self.emission_matrix = emission_matrix
        self.rng = rng if rng is not None else np.random.default_rng()

        if transition_matrix is None and emission_matrix is None: 
            self.init_random(n_symbols, n_states)

        if emission_matrix is not None: 
            self.n_symbols = emission_matrix.shape[1]
        else: 
            self.n_symbols = len(self.transition_matrix)
        
        self.check_probabilities()
        self.symbols = ALPHABET[:self.n_symbols]

    def random_probability_matrix(self, n_rows, n_cols):
        matrix = self.rng.random((n_rows, n_cols))  
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        return matrix
    
    def get_init(self): 
        return self.transition_matrix, self.emission_matrix
    
    def init_random(self, n_symbols, n_states):
        if n_states is None and n_symbols is not None: 
            self.transition_matrix = self.random_probability_matrix(n_symbols, n_symbols)
            self.emission_matrix = None
        elif n_symbols is not None and n_states is not None: 
            self.transition_matrix = self.random_probability_matrix(n_states, n_states)
            self.emission_matrix = self.random_probability_matrix(n_states, n_symbols)
        else: 
            raise ValueError("You must at least specify `n_states` and `n_symbols` for random initialization")
        
        self.n_symbols = n_symbols
        self.n_states = n_states
        return self.transition_matrix, self.emission_matrix

    def generate_length(self): 
        if isinstance(self.sequence_length, tuple) or isinstance(self.sequence_length, list): 
            length = self.rng.integers(low=self.sequence_length[0], high=self.sequence_length[1])
            return length
        elif isinstance(self.sequence_length, int):
            return self.sequence_length 
        else: 
            raise ValueError(f"The `sequence_length` attribute is of type int or tuple or list ")
            
    def check_probabilities(self): 
        if self.transition_matrix is not None: 
            for i in range(self.transition_matrix.shape[0]):
                if not np.isclose(np.sum(self.transition_matrix[i]), 1.0):
                    raise ValueError(f"Row {i} of transition_matrix does not sum to 1.")

        if self.emission_matrix is not None and self.transition_matrix is not None:
            if self.emission_matrix.shape[0] != self.transition_matrix.shape[0]:
                raise ValueError("emission matrix should have same row number as transition matrix.")
            for i in range(self.emission_matrix.shape[0]):
                if not np.isclose(np.sum(self.emission_matrix[i]), 1.0):
                    raise ValueError(f"Row {i} of hidden_matrix does not sum to 1.")
            self.n_hidden = self.transition_matrix.shape[0]
            if self.emission_matrix.shape[0] != self.transition_matrix.shape[0]:
                raise ValueError("Number of hidden states does not match the dimension of transition_matrix.")
            
    def generate_sequence(self, initial_state=None): 
        if initial_state is None:
            current_state = self.rng.choice(self.n_symbols)
        else:
            current_state = initial_state

        sequence = [self.symbols[current_state]]
        l = self.generate_length()
        for _ in range(l - 1):
            next_state = self.rng.choice(self.n_symbols, p=self.transition_matrix[current_state])
            sequence.append(self.symbols[next_state])
            current_state = next_state
        return sequence
    
    def generate_hidden_sequence(self, initial_state=None):       
        if initial_state is None:
            current_hidden_state = self.rng.choice(self.n_hidden)
        else:
            current_hidden_state = initial_state

        current_symbol = self.rng.choice(self.n_symbols, p=self.emission_matrix[current_hidden_state])
        sequence = [self.symbols[current_symbol]]

        l = self.generate_length()
        for _ in range(l - 1):
            next_hidden_state = self.rng.choice(self.n_hidden, p=self.transition_matrix[current_hidden_state])
            emitted_symbol = self.rng.choice(self.n_symbols, p=self.emission_matrix[next_hidden_state])
            sequence.append(self.symbols[emitted_symbol])
            current_hidden_state = next_hidden_state

        return sequence

    def generate_all_sequences(self, n_sequence, initial_state=None):
        all_seqs = []
        for _ in range(n_sequence):
            if self.emission_matrix is not None: 
                seq = self.generate_hidden_sequence(initial_state=initial_state)
            else: 
                seq = self.generate_sequence(initial_state=initial_state)
            all_seqs.append(seq)
        return all_seqs