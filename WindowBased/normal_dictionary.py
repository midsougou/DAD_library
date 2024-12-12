from .window_struct import WindowStruct

class NormalDictionary(WindowStruct): 
    def __init__(self, dataset, window_length, mode="average"): 
        super().__init__(dataset, window_length, mode=mode)

    def train(self): 
        self.frequency_dictionary = {}
        for partition in self.dataset: 
            for sequence in partition: 
                sequence = tuple(sequence)
                if sequence in self.frequency_dictionary.keys():
                    self.frequency_dictionary[tuple(sequence)] = self.frequency_dictionary.get(sequence) + 1

        return self.frequency_dictionary

    def predict(self, test_dataset):
        test_dataset = self.partition(test_dataset) 

        anomaly_scores = []
        for partition in test_dataset: 
            anomalies = []
            for sequence in partition: 
                anomalies.append(self.frequency_dictionary.get(tuple(sequence)))
            
            score = self.process_anomaly(anomalies)
            anomaly_scores.append(score)

        return anomaly_scores