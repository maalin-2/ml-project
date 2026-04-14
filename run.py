import experiment as exp


def todo():
    raise NotImplementedError(
        "Plug in your SGD learner here: todo(training_set, sigma) -> predictor"
    )


def todo_two():
    raise NotImplementedError(
        "Plug in your predictor evaluation here: "
        "todo_two(predictor, testing_set) -> (risk, classification_error)"
    )


# Different parameter settings of n, sigma
parameters = [[50, .2], [50, .4], 
              [100, .2], [100, .4], 
              [500, .2], [500, .4], 
              [1000, .2], [1000, .4]]

def main():
    settings = {0.2: [], 0.4: []}

    for n, sigma in parameters:
        predictor_risks, predictor_bc_errors = exp.run_experiment(
            n, sigma, todo, todo_two
        )
        settings[sigma].append((n, predictor_risks, predictor_bc_errors))

    for sigma, rows in settings.items():
        exp.print_results(sigma, rows)


if __name__ == "__main__":
    main()
