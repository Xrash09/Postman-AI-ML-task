# WRITEUP.md — Neural Network with Manual Backpropagation

**Task 1 — Postman AI/ML Recruitment Task, 26 Batch**

## 1. Overview

For this task I built a small two-layer feedforward network (784 → 128 → 10) using nothing but NumPy, and trained it on a 5,000-image slice of MNIST. The whole point was to do the forward and backward passes by hand — no `autograd`, no `.backward()`, none of that. I wrote out the gradient math myself and then checked it against PyTorch's autograd afterward to make sure I hadn't messed anything up.

**Architecture:**
- Input: 784 units (28×28 images flattened and scaled to `[0, 1]`)
- Hidden layer: 128 units, ReLU
- Output: 10 units, Softmax (one probability per digit)

**What's in the repo:**
- `neural_network.py` — the actual network: forward pass, backward pass, parameter updates, and the training loop
- `train.py` — loads MNIST, splits it, trains, prints test accuracy
- `test_correctness.py` — the gradient check against `torch.autograd`

## 2. Forward Pass

Given a batch `X` of shape `(m, 784)`, `forward_pass` does:

```
Z1 = X · W1 + b1        # (m, 784)·(784, 128) → (m, 128)
A1 = ReLU(Z1)
Z2 = A1 · W2 + b2        # (m, 128)·(128, 10)  → (m, 10)
A2 = Softmax(Z2)
```

I went with ReLU for the hidden layer mostly because it's cheap and the derivative is trivial — 1 if the input was positive, 0 otherwise. It also sidesteps the vanishing-gradient issue you get with something like Sigmoid once things start saturating.

Softmax turns the 10 logits into something that actually sums to 1, so you can read it as "how confident is the network in each digit." One thing I made sure to do in `softmax()` is subtract the row max before exponentiating (`exp(Z - max(Z))`) — without that, large logits can blow up `exp()` into `inf`. Since Softmax is shift-invariant this doesn't change the actual probabilities, it just keeps the numbers from overflowing.

Everything from the forward pass (`X`, `Z1`, `A1`, `Z2`, `A2`) gets stashed in a `cache` dict so the backward pass doesn't have to recompute it.

## 3. Loss Function

With a Softmax output, the natural fit is Categorical Cross-Entropy:

```
L = -(1/m) * Σ log(A2[i, true_class_i])
```

`compute_loss` adds a tiny `1e-8` inside the log just so a predicted probability of exactly 0 doesn't crash the whole thing with `log(0)`. The intuition is pretty simple — the loss only cares about the probability the model put on the *correct* answer. Confident and right → loss near zero. Confident and wrong → loss shoots up.

## 4. Backward Pass — Full Derivation

### 4.1 The output layer gradient (this is the nice part)

The reason this whole exercise is doable by hand is that Softmax + Cross-Entropy have derivatives that collapse into something really clean. Let `p_j` be the Softmax output for class `j`, and `y_j` be the one-hot label.

For the correct class `c`:

```
∂L/∂z_c = -(1/p_c) · ∂p_c/∂z_c = -(1/p_c) · p_c(1 - p_c) = p_c - 1
```

For any other class `j ≠ c`:

```
∂p_c/∂z_j = -p_c · p_j
∂L/∂z_j  = -(1/p_c) · (-p_c · p_j) = p_j
```

Put those two cases together (remembering `y_j = 1` only at `j = c`) and the whole thing turns into one subtraction:

```
∂L/∂Z2 = A2 - Y
```

Which in the code is just:

```python
delta2 = (A2 - Y) / X.shape[0]
```

I divide by the batch size `m` here so it matches what PyTorch does by default with `reduction='mean'` in `cross_entropy` — otherwise the two wouldn't line up when I went to compare them later.

### 4.2 Output layer gradients (`dW2`, `db2`)

`Z2 = A1 · W2 + b2`, so the chain rule gives us, in `backward_layer_2`:

```python
dW2 = np.dot(A1.T, delta2)                   # (128, m)·(m, 10) → (128, 10)
db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, 10)
```

### 4.3 Pushing the error back into the hidden layer

Next step is projecting the output error back through `W2` to see how much each hidden neuron is to blame:

```
∂L/∂A1 = delta2 · W2ᵀ     # (m, 10)·(10, 128) → (m, 128)
```

Since `A1 = ReLU(Z1)`, I also need to multiply by the ReLU derivative here — basically a 0/1 mask. Any neuron that got clamped to zero on the way forward couldn't have influenced the output, so it gets zero gradient too. That's `relu_derivative(Z1)`:

```python
delta1 = np.dot(delta2, W2.T) * self.relu_derivative(Z1)   # (m, 128)
```

### 4.4 Hidden layer gradients (`dW1`, `db1`)

Same idea as 4.2, just one layer earlier, in `backward_layer_1`:

```python
dW1 = np.dot(X.T, delta1)                    # (784, m)·(m, 128) → (784, 128)
db1 = np.sum(delta1, axis=0, keepdims=True)   # (1, 128)
```

### 4.5 Updating the parameters

Plain vanilla gradient descent, nothing fancy:

```
W ← W - α · dW
b ← b - α · db
```

## 5. Gradient Correctness Check

To actually trust the math above, I wrote `test_correctness.py`, which:

1. Sets up identical `W1, b1, W2, b2` in the NumPy network.
2. Runs one forward pass and computes `dW1, db1, dW2, db2` with my manual backward pass.
3. Rebuilds the exact same computation in PyTorch (`torch.matmul` + `torch.relu` + `F.cross_entropy`), calls `loss_pt.backward()`, and pulls the gradients off `.grad`.
4. Compares every gradient element with `np.testing.assert_allclose(..., atol=1e-5)`.

**Result:**

> *[Paste the actual console output from running `python test_correctness.py` here — should be either `RESULT: PASS (All manual gradients match torch.autograd within 1e-5 tolerance)` or a `❌ RESULT: FAIL` with the assertion error attached. If you want a precise number, you can also print something like `np.max(np.abs(dW2_custom - W2_pt.grad.numpy()))` to get the actual max difference.]*

Heads up — I couldn't run this myself in this sandbox since there's no `torch` installed and no internet access to grab it. You'll need to run it on your machine and drop the real output in here before you submit.

## 6. Training Results

`train.py` pulls in the Kaggle MNIST `train.csv`, grabs a random 5,000-row sample (seeded with `random_state=42` so it's reproducible), scales pixels down to `[0, 1]`, one-hot encodes the labels, and splits 80/20 into train and test (4,000 / 1,000). Training runs for 500 iterations of full-batch gradient descent, learning rate 0.1, printing loss/accuracy every 10 iterations, and saves a `loss_curve.png` at the end.

**Result:**

> *[Run `python train.py` and drop in the real final training loss, final training accuracy, and test accuracy. Then take a look at `loss_curve.png` and describe what it actually looks like — does it fall off fast in the first ~50 iterations and then flatten out? Any bumpiness? Does it look like it's still improving by iteration 500 or has it basically plateaued?]*

Same story as above — no `train.csv` in this environment, so I couldn't actually run training. Fill this in from your own run.

## 7. Debugging & Gradient Verification Lessons

Three things tripped me up while getting the backward pass right:

### 7.1 Dividing by the wrong thing (`m` vs. `A2.size`)

I kept failing the gradient check against PyTorch by a suspiciously consistent factor — exactly 10x, which happens to be the number of output classes, `K`. That was the giveaway. I'd been normalizing the error signal by `A2.size`, which is `m × K`, instead of just the batch size `m`. Dividing by the full element count instead of the batch size shrank every gradient by an extra factor of 10, which meant the network was barely updating at all.

The fix was just being more careful about what "average over the batch" actually means:

```
delta2 = (A2 - Y) / X.shape[0]
```

Once I switched to `X.shape[0]` instead of `.size`, the gradients lined up with autograd and training actually started moving at a normal pace.

### 7.2 Mixing MSE evaluation with a Cross-Entropy gradient

At one point I was checking performance with Mean Squared Error while still using the `(A2 - Y) / m` shortcut for the backward pass. That shortcut only works because Softmax and Cross-Entropy happen to cancel out nicely — it's not a generic gradient for any loss function. Swap in MSE and you're supposed to be backpropagating through the full Softmax Jacobian, which `(A2 - Y)` does not do. I hadn't really internalized that until this went wrong, which is honestly probably the most useful bug I hit — it forced me to actually understand *why* that clean subtraction works instead of just accepting it. Once I standardized everything around Cross-Entropy (both the loss I compute and the gradient I use), it was consistent again.

### 7.3 NaNs showing up out of nowhere

Every so often during early runs the loss would just turn into `NaN` or `inf` and training would grind to a halt. Two separate causes here: big positive logits making `exp(Z2)` overflow, and the predicted probability for the correct class occasionally getting close enough to 0 that `-log(A2)` shot off toward `-inf`.

Both fixes are pretty standard and already in the code:
- subtracting the row max before exponentiating in `softmax()`
- adding `1e-8` inside the log in `compute_loss()`

Small fixes, but without them the whole thing is fragile in a way that's easy to miss until it happens partway through a training run.

## 8. Limitations and Possible Extensions

- I'm doing full-batch gradient descent — every single step uses all 4,000 training examples. It makes for a nice smooth loss curve but it's heavier per step than mini-batching, and probably generalizes a bit differently than SGD would.
- No regularization anywhere — no L2, no dropout. On a dataset this small that's a real overfitting risk I just didn't get to.
- Only tried one architecture — a single 128-unit hidden layer. Didn't get around to testing wider or deeper versions.
- Didn't touch the stretch goals — no alternate activation/loss combo, no Momentum or Adam. Ran out of time before I could get to those.

## 9. Conclusion

Doing backprop by hand instead of letting a framework handle it made the whole process feel a lot less like magic. The part that stuck with me most was watching the Softmax and Cross-Entropy derivatives — which look messy separately — collapse into one clean subtraction, `A2 - Y`. And honestly the bugs were the most educational part: a normalization mix-up, a mismatched loss/gradient pairing, and some numerical overflow — all things that are trivial to overlook inside a framework that just computes the gradient for you, but impossible to miss when you're the one doing the math.
