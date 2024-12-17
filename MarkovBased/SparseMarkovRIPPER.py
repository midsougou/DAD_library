from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .markov_struct import MarkovStruct
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SparseMarkovRIPPER(MarkovStruct):
    """Rule-based technique using a decision tree to approximate the RIPPER algorithm."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        self.label_encoder = LabelEncoder()

    def extract_features_and_labels(self, sequences):
        """Extract (context, label) pairs from sliding windows of the training sequences."""
        X, y = [], []
        for sequence in sequences:
            for i in range(len(sequence) - self.max_depth + 1):
                context = sequence[i:i + self.max_depth - 1]
                label = sequence[i + self.max_depth - 1]
                X.append(context)
                y.append(label)
        # Convert X and y from strings to numeric using LabelEncoder
        flat_context = [symbol for context in X for symbol in context]
        self.label_encoder.fit(flat_context + y)  # Fit on both contexts and labels
        X_encoded = np.array([[self.label_encoder.transform([c])[0] for c in context] for context in X])
        y_encoded = self.label_encoder.transform(y)
        return X_encoded, y_encoded

    def _train(self, sequences):
        X, y = self.extract_features_and_labels(sequences)
        self.classifier.fit(X, y)

    def predict(self, sequences):
        logging.debug(f"Input sequences shape: {len(sequences)}")
        contexts_encoded = []
        for sequence in sequences:
            for i in range(len(sequence) - self.max_depth + 1):
                context = sequence[i:i + self.max_depth - 1]  # Extract context of length (max_depth - 1)
                encoded_seq = self.label_encoder.transform(list(context))  # Transform context to numeric encoding
                contexts_encoded.append(encoded_seq)
        
        contexts_encoded = np.array(contexts_encoded)
        logging.debug(f"Shape of contexts_encoded for prediction: {contexts_encoded.shape}")
        predictions = self.classifier.predict(contexts_encoded)
        decoded_predictions = self.label_encoder.inverse_transform(predictions)  # Decode back to original symbols
    
        return decoded_predictions
        # predictions = []
        # for seq in context:
        #     # Transform each sequence individually
        #     context_encoded = np.array(self.label_encoder.transform(seq)).reshape(1, -1)
        #     prediction_index = self.classifier.predict(context_encoded)[0]
        #     predicted_symbol = self.label_encoder.inverse_transform([prediction_index])[0]
        #     predictions.append(predicted_symbol)
        # return predictions
    def compute_anomaly_score(self, sequence):
        log_probability = 0.0
        for i in range(len(sequence) - self.max_depth + 1):
            context = sequence[i:i + self.max_depth - 1]
            symbol = sequence[i + self.max_depth - 1]
            predicted_symbol = self.predict([context])[0]
            if predicted_symbol == symbol:
                prob = 0.9  # Assume high confidence for correct predictions
            else:
                prob = 0.001  # Assume low confidence for incorrect predictions
            log_probability += np.log(prob)
        return -log_probability  # Use -log(P) as the anomaly score
 