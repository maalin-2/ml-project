import dataGeneration as dg
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# This function conducts all 30 experiment runs for the setting of
# n and sigma. It returns the predictor (ŵ) risk and classification error
def run_experiment(n, sigma, testing_set, predictor_generator, predictor_test):
    predictor_risks = []
    bc_errors = []

    for _ in range(30):
        training_set = dg.generate_training_set(n, sigma)
        predictor = predictor_generator(training_set, sigma)
        risk, bc_error = predictor_test(predictor, testing_set)
        predictor_risks.append(risk)
        bc_errors.append(bc_error)

    return predictor_risks, bc_errors


# This function returns the necessary statistics for the experiment
# Minimum, mean, standard deviation, expected excess risk = mean - minimum
def calculate_statistics(data):
    values = np.asarray(data, dtype=float)
    minimum = float(np.min(values))
    mean = float(np.mean(values))
    exp_excess_risk = mean - minimum
    std = float(np.std(values))
    return minimum, mean, std, exp_excess_risk

def plot_graph(sigma, summaries):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    n_values = [item["n"] for item in summaries]
    excess_risk_means = [item["risk_excess"] for item in summaries]
    excess_risk_stds = [item["risk_std"] for item in summaries]
    error_means = [item["error_mean"] for item in summaries]
    error_stds = [item["error_std"] for item in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(
        n_values,
        excess_risk_means,
        yerr=excess_risk_stds,
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    axes[0].set_title(f"Expected Excess Risk (sigma={sigma})")
    axes[0].set_xlabel("Training examples n")
    axes[0].set_ylabel("E[L(w_hat; D)] - min L(w; D)")
    axes[0].set_xticks(n_values)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        n_values,
        error_means,
        yerr=error_stds,
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="tab:orange",
    )
    axes[1].set_title(f"Expected Classification Error (sigma={sigma})")
    axes[1].set_xlabel("Training examples n")
    axes[1].set_ylabel("E[err(w_hat; D)]")
    axes[1].set_xticks(n_values)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"SGD Performance Summary (sigma={sigma})")
    fig.tight_layout()

    output_path = results_dir / f"sgd_results_sigma_{str(sigma).replace('.', '_')}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _build_summary_row(n, predictor_risks, bc_errors):
    risk_min, risk_mean, risk_std, risk_excess = calculate_statistics(predictor_risks)
    _, error_mean, error_std, _ = calculate_statistics(bc_errors)
    return {
        "n": n,
        "risk_min": risk_min,
        "risk_mean": risk_mean,
        "risk_std": risk_std,
        "risk_excess": risk_excess,
        "error_mean": error_mean,
        "error_std": error_std,
    }


def print_results(sigma, setting_rows):
    summaries = []
    for n, predictor_risks, bc_errors in sorted(setting_rows, key=lambda x: x[0]):
        summaries.append(_build_summary_row(n, predictor_risks, bc_errors))

    print(f"\n=== SGD Results for sigma={sigma} ===")
    print(
        "n\tmin_risk\tmean_risk\tstd_risk\texcess_risk\tmean_cls_error\tstd_cls_error"
    )
    for row in summaries:
        print(
            f"{row['n']}\t"
            f"{row['risk_min']:.6f}\t"
            f"{row['risk_mean']:.6f}\t"
            f"{row['risk_std']:.6f}\t"
            f"{row['risk_excess']:.6f}\t"
            f"{row['error_mean']:.6f}\t"
            f"{row['error_std']:.6f}"
        )

    figure_path = plot_graph(sigma, summaries)
    print(f"Saved plot: {figure_path}")