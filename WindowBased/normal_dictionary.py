from .window_struct import WindowStruct

class NormalDictionary(WindowStruct): 
    def __init__(self, window_length, mode="average"): 
        super().__init__(window_length, mode=mode)

    def train(self, dataset): 
        dataset = self.partition(dataset)

        self.frequency_dictionary = {}
        for sequence in dataset: 
            for window in sequence: 
                window = tuple(window)
                self.frequency_dictionary[window] = self.frequency_dictionary.get(window, 0) + 1

        return self.frequency_dictionary
    
    def compute_anomaly_score(self, sequence): 
        anomalies = []
        for window in sequence:
            anomalies.append(self.frequency_dictionary.get(tuple(window), 0)) 

        return anomalies
