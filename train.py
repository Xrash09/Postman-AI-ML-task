import numpy as np
import pandas as pd
from neural_network import NeuralNetwork, gradient_descent, get_predictions, get_accuracy

if __name__ == "__main__":
    np.random.seed(42)

    # Load MNIST dataset from Kaggle CSV
    df = pd.read_csv('train.csv')

    # Use 5000 samples — reduced dataset as suggested in task
    df = df.sample(5000, random_state=42).reset_index(drop=True)

    # Separate pixels and labels, normalize to [0, 1]
    y = df['label'].values
    X = df.drop('label', axis=1).values / 255.0

    # One-hot encode labels into a (5000, 10) matrix.
    # Uses advanced NumPy integer array indexing: np.arange selects every row, 
    # and 'y' selects the specific column index for the true label to set to 1.0.
    Y = np.zeros((len(y), 10))
    Y[np.arange(len(y)), y] = 1.0

    # 80/20 train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Train network — 500 iterations, learning rate 0.1
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.1)

    # Evaluate on unseen test data
    nn = NeuralNetwork()
    A2_test, _ = nn.forward_pass(X_test, W1, b1, W2, b2)
    test_acc = get_accuracy(get_predictions(A2_test), Y_test) * 100
    print(f"\nTest Accuracy: {test_acc:.2f}%")