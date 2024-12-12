import numpy as np 

class WindowStruct: 
    def __init__(self, dataset, window_length, mode="average"): 
        self.window_length = window_length
        self.dataset = self.partition(dataset)
        self.mode = mode

    def partition(self, dataset): 
        partition = []
        for sequence in dataset: 
            partition = [sequence[i*self.window_length:(i+1)*self.window_length] for i in range(int(len(sequence) // self.window_length))]
            partition.append(partition)
        
        return partition

    def process_anomaly(self, anomaly_scores): 
        if self.mode == "average": 
            return np.mean(anomaly_scores)
        elif self.mode == "max": 
            return np.max(anomaly_scores)