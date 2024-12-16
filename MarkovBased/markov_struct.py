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
        labels = np.where(np.abs(scores) > self.bound, 1, 0)
        return np.array(labels)
    
    def train(self, dataset): 
        self._train(dataset)
        proba = self.predict_proba(dataset)
        self.bound = np.percentile(proba, 95)
    
    def reset(self): 
        """method to reset attributes of a lower class"""
        max_depth = self.max_depth
        for attr in vars(self):  
            setattr(self, attr, None)
        self.__init__(max_depth=max_depth)