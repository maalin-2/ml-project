import numpy as np

M = 1
RHO = np.sqrt(2)
D = 5

def euclidean_projection(v):
    return v / np.linalg.norm(v)

# gradient of logistic loss
# z is [label, feature_vector] as produced by dataGeneration
def gradient(w, z):
    y, x = z[0], z[1]
    margin = y * np.dot(w, x)
    return - (y * x * np.exp(-margin)) / (1 + np.exp(-margin))

# examples is a list of [label, feature_vector] with n or more entries
def gradient_descent(examples, T):
    alpha = M / (RHO * np.sqrt(T))
    w = np.zeros((T + 1, D))
    for t in range(T):
        z = examples[t % len(examples)]
        next_wt = w[t] - alpha * gradient(w[t], z)
        if np.linalg.norm(next_wt) > 1:
            next_wt = euclidean_projection(next_wt)
        w[t + 1] = next_wt
    w_hat = 1/T * np.sum(w[1:T+1], axis=0)
    return w_hat

