import numpy as np
from kmedoids import KMedoids

class KernelBased: 
    def __init__(self, dataset, similarity_metric): 
        self.dataset = dataset
        self.similarity_metric = similarity_metric
        self.similarity_matrix = None

    def compute_similarity_matrix(self, n_clusters): 
        n = len(self.dataset)
        self.similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.similarity_metric(self.dataset[i], self.dataset[j])
                self.similarity_matrix[i,j] = sim
                self.similarity_matrix[j,i] = sim  # Symmetric
        
        self.distance_matrix = 1 - self.similarity_matrix
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
        kmedoids.fit(self.distance_matrix)
        self.medoids = self.dataset[kmedoids.cluster_centers_]
        return self.similarity_matrix

    def knearest_predict(self, test_sequence, k_nearest=5):
        similarities = []
        for sequence in self.dataset:  
            similarities.append(self.similarity_metric(test_sequence, sequence))
        
        similarities.sort(reverse=True)
        anomaly_score = 1 / similarities[k_nearest]
        return anomaly_score

    def clustering_predict(self, test_sequence, n_clusters=5):
        if self.similarity_matrix is None: 
            self.compute_similarity_matrix(n_clusters)

        max_similarity = 0
        for medoid in self.medoids: 
            max_similarity = max(max_similarity, self.similarity_metric(test_sequence, medoid))
        
        return 1 / max_similarity     