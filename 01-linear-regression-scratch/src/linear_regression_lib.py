import numpy as np
import matplotlib.pyplot as plt

def z_score_norm(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    eps = 1e-18

    X_norm = (X - mu) / (sigma + eps)

    return X_norm

def compute_function(X, w, b):
    f_x = np.dot(X,w) + b

    return f_x

def compute_cost(X, w, b, y):
    pred = compute_function(X, w, b)
    cost = np.mean((pred-y)**2) / 2

    return cost

def compute_gradient_descent(X, w, b, y, lamda):
    m = X.shape[0]

    pred = compute_function(X, w, b)
    error = pred - y

    dj_w = np.dot(X.T, error) / m + (lamda / m) * w
    dj_b = np.mean(pred - y)


    return dj_w,dj_b

def gradient_descent(X, w, b, y, alpha, lamda=0, iters=100, eps=1e18):
    J_history = []

    for i in range(iters):
        dj_w, dj_b = compute_gradient_descent(X,w,b,y, lamda)

        cost = compute_cost(X, w, b, y)
        J_history.append(cost)

        if eps is not None and len(J_history) > 2:
            if abs(J_history[-2] - J_history[-1]) <= eps:
                break

        w = w - alpha * dj_w
        b = b - alpha * dj_b

    return w, b, J_history

def choose_learning_rate():
    pass

def choose_regularization():
    pass

def model_training():
    pass

def predict():
    pass