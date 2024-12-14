import numpy as np 

class WindowStruct: 
    def __init__(self, window_length, mode="average"): 
        self.window_length = window_length
        self.mode = mode

    def partition(self, dataset): 
        windowed_dataset = []
        symbols = set()
        for sequence in dataset: 
            symbols = symbols.union(set(sequence))
            partition = [sequence[i*self.window_length:(i+1)*self.window_length] for i in range(int(len(sequence) // self.window_length))]
            windowed_dataset.append(partition)
        
        self.n_symbols = len(symbols)
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}
        return windowed_dataset

    def score_sequence(self, anomaly_scores): 
        if self.mode == "average": 
            return np.mean(anomaly_scores)
        elif self.mode == "max": 
            return np.max(anomaly_scores)
        
    def predict_proba(self, dataset): 
        dataset = self.partition(dataset)

        proba = []
        for sequence in dataset: 
            scores = self.compute_anomaly_score(sequence)
            score = self.score_sequence(scores)
            proba.append(score)

        return proba