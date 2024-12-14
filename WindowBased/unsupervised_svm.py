from .window_struct import WindowStruct
from sklearn.svm import OneClassSVM
import numpy as np 

class UnsupervisedSVM(WindowStruct): 
    def __init__(self, window_length, mode="average", gamma='auto'):
        super().__init__(window_length, mode=mode)
        self.gamma = gamma 

    def sequence_encoding(self, sequence): 
        one_hot_encoded = []
        for window in sequence:
            mat = np.zeros((self.n_symbols, len(window)), dtype=int)

            for pos, symbol in enumerate(window):
                s_idx = self.symbol_to_idx[symbol]
                mat[s_idx, pos] = 1

            one_hot_encoded.append(mat.flatten())
        return one_hot_encoded

    def one_hot_encode(self, dataset): 
        one_hot_encoding = []
        for sequence in dataset:
            one_hot = self.sequence_encoding(sequence)
            one_hot_encoding.append(one_hot)
        return np.array(one_hot_encoding)
    
    def train(self, dataset): 
        dataset = self.partition(dataset)
        dataset = self.one_hot_encode(dataset)
        dataset = dataset.reshape(-1, dataset.shape[2])
        self.svm = OneClassSVM(gamma=self.gamma).fit(dataset)
        return self.svm
    
    def compute_anomaly_score(self, sequence): 
        sequence = self.sequence_encoding(sequence)
        return self.svm.predict(sequence) 