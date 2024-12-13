from .kernel_struct import KernelStruct

class KnearestKernel(KernelStruct): 
    def __init__(self, similarity_metric): 
        super().__init__(similarity_metric)

    def predict_sample(self, test_sequence, k_nearest=5):
        similarities = []
        for sequence in self.dataset:  
            similarities.append(self.similarity_metric(test_sequence, sequence))
        
        similarities.sort(reverse=True)
        anomaly_score = 1 / similarities[k_nearest]
        return anomaly_score
    
    def train(self, dataset): 
        self.compute_similarity_matrix(dataset)
    
