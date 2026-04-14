import numpy as np

M = RHO = 1
D = 5

def euclidean_projection(v):
    return v / np.linalg.norm(v)

# gradient of logistic loss
def gradient(w, z):
    x, y = z
    margin = y * np.dot(w, x)
    return - (y * x * np.exp(-margin)) / (1 + np.exp(-margin))

# examples should be (x, y) , n or so ...
def gradient_descent(examples, T):
    alpha = M / (RHO * np.sqrt(T))
    w = np.zeros((T + 1, D))
    for t in range(1, T):
        cur_wt = w[t]
        z = examples[t]
        next_wt = cur_wt - alpha * gradient(cur_wt, z)
        if np.linalg.norm(next_wt) > 1:
            next_wt = euclidean_projection(next_wt)
        w[t + 1] = next_wt
    w_hat = 1/T * np.sum(w[:T], axis = 0)
    return w_hat

