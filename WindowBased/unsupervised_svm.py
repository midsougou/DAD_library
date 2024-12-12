from .window_struct import WindowStruct
import numpy as np 

class UnsupervisedSVM(WindowStruct): 
    def __init__(self, dataset, window_length, mode="average"):
        super().__init__(dataset, window_length, mode=mode)
        self.dataset = self.one_hot_encoding(dataset)

    def one_hot_encoding(self): 
        new_dataset = []
        for partition in self.dataset:
            one_hot_encoded = []
            for seq in partition:
                mat = np.zeros((self.n_symbols, len(seq)), dtype=int)

                for pos, symbol in enumerate(seq):
                    s_idx = self.symbol_to_idx[symbol]
                    mat[s_idx, pos] = 1

                one_hot_encoded.append(mat)
            new_dataset.append(one_hot_encoded)
        return new_dataset
    
    def train(self): 
        self.svm = OneClassSVM(gamma='auto').fit(self.dataset)
        return self.svm
    
    def predict(self, test): 
        test = self.partition(test)
        test = self.one_hot_encoding(test)
        return self.svm.predict(test) 