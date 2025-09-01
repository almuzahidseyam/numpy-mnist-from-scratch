import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Dataset Loading (Assuming Kaggle Format)
# ==========================================
def load_data(filepath):
    """
    Loads data from CSV. Expects Kaggle MNIST format:
    Column 0: Label, Columns 1-784: Pixels.
    """
    data = pd.read_csv(filepath)
    data = np.array(data)
    np.random.seed(42)
    np.random.shuffle(data)
    return data

# ==========================================
# 2. Activation Functions
# ==========================================
def ReLu(Z):
    return np.maximum(Z, 0)

def deriv_ReLu(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    # Subtracting max for numerical stability
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# ==========================================
# 3. Neural Network Functions
# ==========================================
def forward_prop(X, theta1, theta2, theta3, theta4):
    m = X.shape[1]
    
    # Layer 1
    Z2 = theta1.dot(X)
    A21 = ReLu(Z2)
    A2 = np.ones((theta2.shape[1], m))
    A2[1:, :] = A21

    # Layer 2
    Z3 = theta2.dot(A2)
    A31 = ReLu(Z3)
    A3 = np.ones((theta3.shape[1], m))
    A3[1:, :] = A31

    # Layer 3
    Z4 = theta3.dot(A3)
    A41 = ReLu(Z4)
    A4 = np.ones((theta4.shape[1], m))
    A4[1:, :] = A41

    # Output Layer
    Z5 = theta4.dot(A4)
    A5 = softmax(Z5)

    return A2, Z2, A3, Z3, A4, Z4, A5, Z5

def Y_convert(Y):
    Y_encoded = np.zeros((Y.size, Y.max() + 1))
    Y_encoded[np.arange(Y.size), Y.astype(int)] = 1
    return Y_encoded.T

def back_prop(X, Y, A2, Z2, A3, Z3, A4, Z4, A5, theta1, theta2, theta3, theta4, alpha):
    m = Y.shape[1]

    dZ5 = A5 - Y
    dtheta4 = (1/m) * dZ5.dot(A4.T)

    dA4 = theta4.T.dot(dZ5)
    dZ4 = dA4[1:, :] * deriv_ReLu(Z4)
    dtheta3 = (1/m) * dZ4.dot(A3.T)

    dA3 = theta3.T.dot(dZ4)
    dZ3 = dA3[1:, :] * deriv_ReLu(Z3)
    dtheta2 = (1/m) * dZ3.dot(A2.T)

    dA2 = theta2.T.dot(dZ3)
    dZ2 = dA2[1:, :] * deriv_ReLu(Z2)
    dtheta1 = (1/m) * dZ2.dot(X.T)

    # Update weights
    theta1 -= alpha * dtheta1
    theta2 -= alpha * dtheta2
    theta3 -= alpha * dtheta3
    theta4 -= alpha * dtheta4

    return theta1, theta2, theta3, theta4

def predict(X, theta1, theta2, theta3, theta4):
    _, _, _, _, _, _, A5, _ = forward_prop(X, theta1, theta2, theta3, theta4)
    return np.argmax(A5, axis=0)

# ==========================================
# 4. Training Loop (Mini-batch)
# ==========================================
def train_network(X_train, Y_train, X_test, Y_test, n_input=784, hidden_1=128, hidden_2=64, hidden_3=32, output_dim=10, alpha=0.01, epochs=50, batch_size=64):
    m_train = X_train.shape[1]
    
    # He/Xavier Initialization
    theta1 = np.random.randn(hidden_1, n_input + 1) * np.sqrt(2./n_input)
    theta2 = np.random.randn(hidden_2, hidden_1 + 1) * np.sqrt(2./hidden_1)
    theta3 = np.random.randn(hidden_3, hidden_2 + 1) * np.sqrt(2./hidden_2)
    theta4 = np.random.randn(output_dim, hidden_3 + 1) * np.sqrt(2./hidden_3)

    Y_train_encoded = Y_convert(Y_train)
    
    for epoch in range(epochs):
        perm = np.random.permutation(m_train)
        X_train_shuff = X_train[:, perm]
        Y_train_shuff = Y_train_encoded[:, perm]

        for start in range(0, m_train, batch_size):
            end = start + batch_size
            X_batch = X_train_shuff[:, start:end]
            Y_batch = Y_train_shuff[:, start:end]

            A2, Z2, A3, Z3, A4, Z4, A5, Z5 = forward_prop(X_batch, theta1, theta2, theta3, theta4)
            theta1, theta2, theta3, theta4 = back_prop(
                X_batch, Y_batch, A2, Z2, A3, Z3, A4, Z4, A5, theta1, theta2, theta3, theta4, alpha
            )

        if (epoch + 1) % 5 == 0:
            _, _, _, _, _, _, A5_train, _ = forward_prop(X_train, theta1, theta2, theta3, theta4)
            loss = -np.mean(np.sum(Y_train_encoded * np.log(A5_train + 1e-8), axis=0))
            train_acc = np.mean(np.argmax(A5_train, axis=0) == Y_train)
            test_acc = np.mean(predict(X_test, theta1, theta2, theta3, theta4) == Y_test)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

    return theta1, theta2, theta3, theta4

if __name__ == "__main__":
    # Note: Replace with actual path to train.csv
    print("Please ensure train.csv is in the correct directory.")
    # data = load_data('train.csv')
    # ... Preprocessing logic here
