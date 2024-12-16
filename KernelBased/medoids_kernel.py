from .kernel_struct import KernelStruct
import numpy as np 
import kmedoids

class MedoidsKernel(KernelStruct): 
    def __init__(self, similarity_metric): 
        super().__init__(similarity_metric)

    def compute_kemedoids(self, kmax=10, kmin=1): 
        km = kmedoids.dynmsc(self.distance_matrix, kmax, kmin)
        self.medoids = [self.dataset[medoid] for medoid in km.medoids]
        return self.medoids
    
    def set_new_medoids(self, k): 
        km = kmedoids.fastermsc(self.distance_matrix, k)
        self.medoids = [self.dataset[medoid] for medoid in km.medoids]
        return self.medoids

    def predict_sample(self, test_sequence):
        max_similarity = 0
        for medoid in self.medoids: 
            max_similarity = max(max_similarity, self.similarity_metric(test_sequence, medoid))
        
        return 1 / max_similarity   
    
    def train(self, dataset, kmax=10, kmin=1): 
        self.compute_similarity_matrix(dataset)
        self.compute_kemedoids(kmax=kmax, kmin=kmin)


    

    
    
    