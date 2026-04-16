import numpy as np

M = 1
RHO = np.sqrt(2)
D = 5

# projects v onto the unit ball
def euclidean_projection(v):
    return v / np.linalg.norm(v)

# returns the gradient of the logistic loss at w for example z
def gradient(w, z):
    y, x = z[0], z[1]
    margin = y * np.dot(w, x)
    return - (y * x * np.exp(-margin)) / (1 + np.exp(-margin))

# runs projected sgd for T steps and returns the averaged weight vector
def gradient_descent(examples, T):
    alpha = M / (RHO * np.sqrt(T))
    w = np.zeros((T + 1, D))

    # sgd update loop: take a gradient step then project back if outside unit ball
    for t in range(T):
        z = examples[t % len(examples)]
        next_wt = w[t] - alpha * gradient(w[t], z)
        if np.linalg.norm(next_wt) > 1:
            next_wt = euclidean_projection(next_wt)
        w[t + 1] = next_wt

    # average iterates to get final predictor
    w_hat = 1/T * np.sum(w[1:T+1], axis=0)
    return w_hat
