import numpy as np

class MarkovStruct: 
    def __init__(self): 
        pass

    def predict_proba(self, dataset): 
        anomaly = []
        for sequence in dataset:
            score = self.compute_anomaly_score(sequence)
            anomaly.append(score)
            
        return np.array(anomaly)
    
    def predict(self, dataset): 
        scores = self.predict_proba(dataset)
        labels = np.where(np.abs(scores) < self.bound, 1, 0)
        return labels