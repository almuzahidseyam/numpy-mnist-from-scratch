# Neural Network from Scratch in Pure NumPy

This repository contains a 4-layer Neural Network built entirely from scratch using pure `NumPy`. It is designed to classify handwritten digits (MNIST dataset) without relying on heavy machine learning frameworks like PyTorch or TensorFlow.

## ðŸš€ Features
- **Pure NumPy Implementation**: Matrix operations, forward propagation, and backpropagation all manually implemented.
- **Architecture**: 4 layers (Input: 784 -> Hidden 1: 128 -> Hidden 2: 64 -> Hidden 3: 32 -> Output: 10).
- **Activations**: Custom implementation of ReLU for hidden layers and Softmax (with numerical stability tricks) for the output layer.
- **Optimization**: Mini-batch Gradient Descent, He (Xavier) Initialization, and **Dropout Regularization (Inverted Dropout)**.
- **Metrics Tracker**: Tracks Cross-Entropy Loss and accuracy per epoch.

## ðŸ§  Why this project?
Building a neural network from scratch helps in deeply understanding the fundamental mathematics behind deep learning. This includes:
- How gradients flow backwards (Chain Rule).
- Why activation functions like ReLU prevent vanishing gradients.
- How Softmax pairs mathematically with Categorical Cross-Entropy.

## âš™ï¸ How to Run
1. Place the Kaggle MNIST `train.csv` in the root directory.
2. Run the script:
   ```bash
   python train_mnist.py
   ```

## ðŸ› ï¸ Lessons Learned
- Visualizing random permutations to check if predictions are stuck at local minima (e.g., predicting the same class repeatedly due to shuffling/bias issues).
- The importance of matching learning rates (`alpha`) to the weight initialization strategy.
- How memory-efficient mini-batching stabilizes the loss curve.

