from .window_struct import WindowStruct
import numpy as np

class Lookahead(WindowStruct): 
    def __init__(self, window_length, mode="average", k_look_ahead=5): 
        super().__init__(window_length, mode=mode)
        self.k_look_ahead = k_look_ahead
    
    def train(self, dataset):
        self.dataset = self.partition(dataset) 

        self.lookahead_dict = {}
        for sequence in self.dataset:
            for window in sequence: 
                for i in range(self.window_length - self.k_look_ahead):
                    pair = (window[i], window[i + self.k_look_ahead])
                    self.lookahead_dict[pair] = self.lookahead_dict.get(pair, 0) + 1
        return self.lookahead_dict
    
    def compute_anomaly_score(self, sequence): 
        scores = []
        for window in sequence: 
            anomalies = []
            for i in range(len(window) - self.k_look_ahead):
                pair = (window[i], window[i + self.k_look_ahead])
                anomalies.append(self.lookahead_dict.get(pair, 0))
            scores.append(np.mean(anomalies))

        return scores