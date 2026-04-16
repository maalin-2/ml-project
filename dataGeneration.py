import random
import numpy as np

# generates a single labeled example [label, feature_vector] with gaussian noise scaled by sigma
def generate_vector(sigma):
    y = random.randint(0,1)
    mu = np.array([1/4, 1/4, 1/4, 1/4])

    label = 1
    if (y == 0):
        mu = mu * -1
        label = -1

    # sample gaussian noise
    z = np.array([np.random.normal() for _ in range(4)])

    u = mu + sigma * z

    # project onto unit ball if outside
    norm = np.linalg.norm(u)
    if (norm > 1):
        u = u / norm

    u = np.append(u, 1)  # bias term
    return [label, u]

# generates n labeled examples for a given sigma
def generate_training_set(n, sigma):
    training_set = []
    for _ in range(n):
        training_set.append(generate_vector(sigma))

    return training_set

# generates a fixed test set of 400 examples; seeded so it's the same every run
def generate_testing_set(sigma):
    random.seed(sigma)
    np.random.seed(int(sigma * 10))
    return generate_training_set(400, sigma)

if __name__ == "__main__":
    generate_training_set(50, 30)
    generate_testing_set(.4)
