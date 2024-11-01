import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        return 1 if weighted_input >= 0 else -1

    def predict(self, inputs):
        prediction = []
        features_pred = np.c_[np.ones(inputs.shape[0]), inputs]
        for x in features_pred:
            y = self.weighting(x)
            prediction.append(self.activation(y))
        return prediction

    def fit(self, inputs, outputs, learning_rate, epochs):
        features = np.c_[np.ones(inputs.shape[0]), inputs]
        self.weights = np.random.rand(features.shape[1])
        for e in range(epochs):
            for x, y in zip(features, outputs):
                weighted_input = self.weighting(x)
                prediction = self.activation(weighted_input)
                self.weights += learning_rate * (y - prediction) * x
            accuracy = accuracy_score(outputs, self.predict(inputs))
            print(f"Epoch {e+1}/{epochs} - Accuracy: {accuracy:.4f}")
            if accuracy == 1.0:
                break

