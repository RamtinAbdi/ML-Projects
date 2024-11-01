import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        return weighted_input

    def predict(self, inputs):
        predictions = []
        features_pred = np.c_[np.ones(inputs.shape[0]), inputs]
        for x in features_pred:
            weighted_input = self.weighting(x)
            activation_output = self.activation(weighted_input)
            prediction = 1 if activation_output >= 0 else -1
            predictions.append(prediction) 
        return np.array(predictions)

    def fit(self, inputs, outputs, learning_rate=0.1, epochs=64):
        features = np.c_[np.ones(inputs.shape[0]), inputs]
        self.weights = np.random.rand(features.shape[1])
        for e in range(epochs):
            weighted_inputs = self.weighting(features)
            activation_outputs = self.activation(weighted_inputs)
            errors = outputs - activation_outputs
            self.weights += learning_rate * np.dot(features.T, errors)
            accuracy = accuracy_score(outputs, self.predict(inputs))
            print(f"Epoch {e+1}/{epochs} - Accuracy: {accuracy:.4f}")
            if accuracy == 1.0:
                break

