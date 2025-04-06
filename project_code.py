"""
Concrete Strength Predictor using Artificial Neural Networks (ANN)
-------------------------------------------------------------------
This script implements an ANN to predict the 28-day compressive strength
of concrete based on key mix proportions using backpropagation.

Author: [Your Name]
Repository: https://github.com/your-username/concrete-strength-predictor
"""

import math
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import os

# -------------------------------
# Neural Network Class Definition
# -------------------------------

class ConcreteANN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate

        # Xavier weight initialization
        self.w_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Sigmoid activation function
        self.activation = lambda x: scipy.special.expit(x)

    def train(self, inputs_data, targets_data):
        inputs = np.array(inputs_data, ndmin=2).T
        targets = np.array(targets_data, ndmin=2).T

        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        # Update weights
        self.w_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                      hidden_outputs.T)
        self.w_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                      inputs.T)

        return output_errors

    def predict(self, inputs_data):
        inputs = np.array(inputs_data, ndmin=2).T
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs


# --------------------------
# Model Configuration
# --------------------------

INPUT_NODES = 7
HIDDEN_NODES = 70
OUTPUT_NODES = 1
LEARNING_RATE = 0.01
EPOCHS = 1000

# Create model instance
model = ConcreteANN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

# --------------------------
# Training Phase
# --------------------------

rms_losses = []
with open("train_test_data/training_data.csv", 'r') as f:
    training_data = f.readlines()

for epoch in range(EPOCHS):
    errors = []
    for record in training_data:
        values = list(map(float, record.strip().split(',')))
        inputs = values[0:7]
        targets = values[7:]
        output_error = model.train(inputs, targets)
        errors.append(output_error**2)
    mean_error = np.mean([e[0, 0] for e in errors])
    rms_losses.append(math.sqrt(mean_error))

# Plot RMS error over epochs
plt.figure(1)
plt.plot(rms_losses, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Root Mean Square Error")
plt.title("Training Performance")
plt.grid(True)
plt.savefig("results/rms_loss.png")

# --------------------------
# Testing Phase
# --------------------------

with open("train_test_data/testing_data.csv", 'r') as f:
    testing_data = f.readlines()

errors_squared = []
targets_squared = []
predictions = []
actuals = []

for record in testing_data:
    values = list(map(float, record.strip().split(',')))
    inputs = values[0:7]
    targets = values[7:]
    prediction = model.predict(inputs)

    predictions.append(prediction[0][0])
    actuals.append(targets[0])
    errors_squared.append((targets[0] - prediction[0][0])**2)
    targets_squared.append(targets[0]**2)

# Evaluation metrics
rrmse = math.sqrt(np.sum(errors_squared)) / math.sqrt(np.sum(targets_squared))
ce = 1 - rrmse**2
print(f"Relative RMSE: {rrmse:.4f}")
print(f"Model Coefficient of Efficiency (C.E.): {ce:.4f}")

# --------------------------
# Visualization of Results
# --------------------------

# Equality Scatter Plot
plt.figure(2)
plt.scatter(predictions, actuals, color='blue', label='Data')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual (Scatter Plot)')
plt.legend()
plt.grid(True)
plt.savefig("results/predicted_vs_actual.png")

# Line Graph of Predictions and Targets
plt.figure(3)
plt.plot(predictions, color='blue', label='Predicted', marker='o')
plt.plot(actuals, color='red', label='Actual', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Compressive Strength')
plt.title('Predicted vs Actual (Line Plot)')
plt.legend()
plt.grid(True)
plt.savefig("results/model_performance.png")

plt.show()
