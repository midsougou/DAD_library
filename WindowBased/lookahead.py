from .window_struct import WindowStruct

class Lookahead(WindowStruct): 
    def __init__(self, dataset, window_length, mode="average", k_look_ahead=5): 
        super().__init__(dataset, window_length, mode=mode)
        self.k_look_ahead = k_look_ahead
    
    def train(self): 
        self.lookahead_dict = {}
        for partition in self.dataset:
            for seq in partition: 
                for i in range(len(seq) - self.k_look_ahead):
                    pair = (seq[i], seq[i + self.k_look_ahead])
                    self.lookahead_dict[pair] = self.lookahead_dict.get(pair, 0) + 1
        return self.lookahead_dict

    def predict(self, test_dataset): 
        test_dataset = self.partition(test_dataset)

        anomaly_score = []
        for partition in test_dataset: 
            for sequence in partition:
                anomalies = []
                for i in range(len(sequence) - self.k_look_ahead):
                    pair = (sequence[i], sequence[i + self.k_look_ahead])
                    anomalies.append(self.lookahead_dict.get(pair, 0))

            anomaly_score.append()

        return anomaly_score