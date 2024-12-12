import kmedoids 
import numpy as np 

class KernelStruct: 
    def __init__(self, similarity_metric): 
        self.similarity_metric = similarity_metric
        self.similarity_matrix = None
        self.medoids = None

    def compute_similarity_matrix(self, dataset): 
        self.dataset = dataset
        n = len(self.dataset)
        self.similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.similarity_metric(self.dataset[i], self.dataset[j])
                self.similarity_matrix[i,j] = sim
                self.similarity_matrix[j,i] = sim  # Symmetric
        
        self.distance_matrix = 1 - self.similarity_matrix
        return self.similarity_matrix
    
    def predict_proba(self, test_set): 
        scores = []

        for test_sequence in test_set: 
            scores.append(self.predict_sample(test_sequence))
        
        return scores



    
        