import numpy as np
import experiment as exp
import gd


def predictor_generator(training_set, sigma):
    T = len(training_set)
    return gd.gradient_descent(training_set, T)


def predictor_test(predictor, testing_set):
    w = predictor
    total_loss = 0.0
    total_error = 0
    for z in testing_set:
        y, x = z[0], z[1]
        margin = y * np.dot(w, x)
        total_loss += np.log(1 + np.exp(-margin))
        pred = 1 if np.dot(w, x) >= 0 else -1
        total_error += 1 if pred != y else 0
    risk = total_loss / len(testing_set)
    bc_error = total_error / len(testing_set)
    return float(risk), float(bc_error)


# Different parameter settings of n, sigma
parameters = [[50, .2], [50, .4], 
              [100, .2], [100, .4], 
              [500, .2], [500, .4], 
              [1000, .2], [1000, .4]]

def main():
    settings = {0.2: [], 0.4: []}

    for n, sigma in parameters:
        predictor_risks, predictor_bc_errors = exp.run_experiment(
            n, sigma, predictor_generator, predictor_test
        )
        settings[sigma].append((n, predictor_risks, predictor_bc_errors))

    for sigma, rows in settings.items():
        exp.print_results(sigma, rows)


if __name__ == "__main__":
    main()
