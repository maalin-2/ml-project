import random
import numpy as np

# TODO - Comment code
def generate_vector(sigma):
    y = random.randint(0,1)
    mu = np.array([1/4, 1/4, 1/4, 1/4])
    
    label = 1
    if (y == 0):
        mu = mu * -1
        label = -1

    z = np.array([np.random.normal() for _ in range(4)])

    u = mu + sigma * z

    norm = np.linalg.norm(u)
    if (norm > 1):
        u = u / norm

    u = np.append(u, 1)  # bias term
    return [label, u]

# The training set is returned as a list
# Each element of the list is [label, numpy.array(feature vector)]
# n - number of feature vectors
# sigma - parameter for gaussian vector generation
def generate_training_set(n, sigma):
    training_set = []
    for _ in range(n):
        training_set.append(generate_vector(sigma))
    
    return training_set

# This function generates the testing set for sigma = .2 or sigma = .4
# It is hard coded to ensure we generate the same testing set each time
# for the respective sigma we use
def generate_testing_set(sigma):
    random.seed(sigma)
    np.random.seed(int(sigma * 10))
    return generate_training_set(400, sigma)

if __name__ == "__main__":
    generate_training_set(50, 30)
    generate_testing_set(.4)