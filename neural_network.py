import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class NeuralNetwork:


    # defines initial layers 1 and 2 with weights and  biases, using random values,0.01 is also multiplied so the values are small and easy to understand , rather than large values causing random noise
    def init_layer_1(self, input_dim, hidden_dim):
        W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        b1 = np.zeros((1, hidden_dim))
        return W1, b1

    def init_layer_2(self, hidden_dim, output_dim):
        W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        b2 = np.zeros((1, output_dim))
        return W2, b2

    
    # ReLU activation function: outputs Z if Z > 0, else 0. This introduces non-linearity.
    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    # Derivative of ReLU: 1 if Z > 0, else 0. Acts as a gate during backprop.
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    # Softmax converts raw logits into a probability distribution.
    # Subtracting np.max(Z) is a numerical stability trick to prevent np.exp() 
    # from overflowing to infinity without changing the final probability ratios.
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    #Takes input data and feeds it into the network layer by layer to compute final predictions
    def forward_pass(self, X, W1, b1, W2, b2):
        Z1 = np.dot(X, W1) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = self.softmax(Z2)

        #the cache dictionay stores these values to compute the derivates later in the backward propagation
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    # cross-entropy loss: penalises the model when confidence in the correct class is low
    # -log(p) is large when p is small, small when p is close to 1
    # 1e-8 prevents log(0) which would give infinity
    def compute_loss(self, A2, Y):
        m = Y.shape[0]
        log_probs = -np.log(A2[np.arange(m), np.argmax(Y, axis=1)] + 1e-8)
        return np.mean(log_probs)


    #computes the gradients for output of W2 and b2 and computes their net error using dot product and sum , Using the chain rule: dL/dW2 = (dL/dZ2) * (dZ2/dW2) = A1.T (dot) delta2 ,dL/db2 = sum of delta2 across the batch size.
    def backward_layer_2(self, delta2, A1):
        dW2 = np.dot(A1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        return dW2, db2

    #back propagates the error through w2 to compute gradients for hidden layer parameters w1 and b1 done via dot product and summation again , delta1 (Error at hidden layer) = (delta2 * W2.T) element-wise multiplied by ReLU derivative , dL/dW1 = X.T (dot) delta1 , dL/db1 = sum of delta1 across the batch.
    def backward_layer_1(self, delta2, W2, Z1, X):
          delta1 = np.dot(delta2, W2.T)* self.relu_derivative(Z1)
          dW1 =np.dot(X.T, delta1)
          db1 = np.sum(delta1, axis=0, keepdims=True)
          return dW1, db1


    #updates weigths and biases and the main zone where learning takes place , the system learns from its mistakes and updates the parameters to reduce the loss in the next attempt
    def update_parameters(self,W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate):
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2

        return W1, b1 , W2, b2

#defines a function that takes final layer activation matrix and finds column index of maximum value in each row , output neuron with highest value is further chosen as the prediction
def get_predictions(A2): 
    return np.argmax(A2, axis=1) 


#checks he accuracy of the prediction by comparing the labels of the prediction with the actual labels , hence giving the perecent of correct predictions made
def get_accuracy(predictions, Y): 
    if Y.ndim > 1 and Y.shape[1] > 1: 
        Y_labels = np.argmax(Y, axis=1) 
    else: 
        Y_labels = Y 
    return np.mean(predictions == Y_labels) 


#trains the neural network through various iterations , calculates learnign rate to update w and b , and prints the loss and accuracy at every 10th iteration
def gradient_descent(X, Y, iterations, alpha): 
    nn = NeuralNetwork() 
    input_dim = X.shape[1] 
    hidden_dim = 128 
    output_dim = Y.shape[1] 
    loss_history = []

    
    W1, b1 = nn.init_layer_1(input_dim, hidden_dim) 
    W2, b2 = nn.init_layer_2(hidden_dim, output_dim) 
    
    for i in range(iterations): 

        A2, cache = nn.forward_pass(X, W1, b1, W2, b2) 
        delta2 = (A2 - Y) / X.shape[0]
        dW2, db2 = nn.backward_layer_2(delta2, cache["A1"]) 
        dW1, db1 = nn.backward_layer_1(delta2, W2, cache["Z1"], cache["X"]) 
        
        W1, b1, W2, b2 = nn.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) 
        loss_history.append(nn.compute_loss(A2, Y))

        if i % 10 == 0 or i == iterations - 1: 
            loss = nn.compute_loss(A2, Y) 
            acc = get_accuracy(get_predictions(A2), Y) * 100.0 
            print(f"Iteration {i:3d} | Loss: {loss:.6f} | Accuracy: {acc:.2f}%") 


    #used to label the axis and plot the curve for the loss history of the model over the 500 iterations that take place
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss over 500 Iterations")
    plt.savefig("loss_curve.png")

    return W1, b1, W2, b2 

