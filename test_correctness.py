import numpy as np
import torch
from neural_network import NeuralNetwork

def test_gradient_correctness():
    print("--- Running Gradient Correctness Harness ---")
    
    # Fixed random seeds for deterministic output
    np.random.seed(42)
    torch.manual_seed(42)

    nn = NeuralNetwork()
    input_dim, hidden_dim, output_dim = 784, 128, 10
    
    # 1. Initialize parameters
    W1_np, b1_np = nn.init_layer_1(input_dim, hidden_dim)
    W2_np, b2_np = nn.init_layer_2(hidden_dim, output_dim)
    X_np = np.random.randn(1, input_dim)
    y_np = np.array([[1.0] + [0.0]*9])

    # 2. Custom Manual Backpropagation
    A2_np, cache = nn.forward_pass(X_np, W1_np, b1_np, W2_np, b2_np)
    delta2 = (A2_np - y_np) / X_np.shape[0]
    dW2_custom, db2_custom = nn.backward_layer_2(delta2, cache["A1"])
    dW1_custom, db1_custom = nn.backward_layer_1(delta2, W2_np, cache["Z1"], cache["X"])

    #3. PyTorch Autograd Reference
    #Note: torch.nn.functional.cross_entropy applies Softmax internally AND 
    #defaults to reduction='mean'. This is why our manual delta2 divides by 
    #the batch size (X_np.shape[0]) to ensure an exact 1:1 mathematical match.
    X_pt = torch.tensor(X_np, requires_grad=True)
    W1_pt = torch.tensor(W1_np, requires_grad=True)
    b1_pt = torch.tensor(b1_np, requires_grad=True)
    W2_pt = torch.tensor(W2_np, requires_grad=True)
    b2_pt = torch.tensor(b2_np, requires_grad=True)

    Z1_pt = torch.matmul(X_pt, W1_pt) + b1_pt
    A1_pt = torch.relu(Z1_pt)
    Z2_pt = torch.matmul(A1_pt, W2_pt) + b2_pt
    y_index = torch.tensor([np.argmax(y_np)])
    loss_pt = torch.nn.functional.cross_entropy(Z2_pt, y_index)
    loss_pt.backward()


    # Verifying all 4 parameter gradients match torch.autograd within 1e-5
    # dW2, db2 — output layer gradients
    # dW1, db1 — hidden layer gradients
    # 4. Verify gradient match within 1e-5 tolerance
    np.testing.assert_allclose(dW2_custom, W2_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(db2_custom, b2_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(dW1_custom, W1_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(db1_custom, b1_pt.grad.numpy(), atol=1e-5)

    print("RESULT: PASS (All manual gradients match torch.autograd within 1e-5 tolerance)")

if __name__ == "__main__":
    try:
        test_gradient_correctness()
    except Exception as e:
        print(f"❌ RESULT: FAIL - {e}")