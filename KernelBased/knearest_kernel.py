from .kernel_struct import KernelStruct

class KnearestKernel(KernelStruct): 
    def __init__(self, similarity_metric, k_nearest=5): 
        super().__init__(similarity_metric)
        self.past_predictions = {}
        self.k_nearest = k_nearest

    def set_knearest(self, k_nearest): 
        self.k_nearest = k_nearest

    def predict_sample(self, test_sequence):
        """This method stores past predictions for computational efficiency 
        when changing hyperparameters and reevaluating."""
        similarities = []
        if tuple(test_sequence) not in self.past_predictions.keys():
            for sequence in self.dataset:  
                similarities.append(self.similarity_metric(test_sequence, sequence))
            similarities.sort(reverse=True)
            self.past_predictions[tuple(test_sequence)] = similarities
        else: 
            similarities = self.past_predictions[tuple(test_sequence)]
        
        anomaly_score = 1 / similarities[self.k_nearest]
        return anomaly_score
    
    def train(self, dataset): 
        self.dataset = dataset
        pass
    
