import pandas as pd
import xgboost as xgb

import shap
from shap import maskers
import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def benchmark_explainer(
    explainer, data,
    to_try_N=range(100, 2000, 100),
    num_runs=100
):
    data_x = data.drop('y', axis=1)
    run_timesN = []
    Ns = []

    # Iterate over the range of N values
    for N in to_try_N:
        print(f'Running for N={N}...')
        # Function to run SHAP explanation on subset
        def run_explainer():
            explainer(data_x.iloc[:N, :])

        # Measure execution times over num_runs
        run_times = np.array([timeit.timeit(run_explainer, number=1) for _ in range(num_runs)])

        run_timesN.append(run_times)
        Ns.append(N)

    return (Ns, run_timesN)


def get_benchmark_times(benchmark_res, confidence=0.95):
    times_mean = []
    times_median = []
    times_ci = []
    Ns = benchmark_res[0]
    run_timesN = benchmark_res[1]

    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)  # Z-value for 95% confidence

    for N, run_time in zip(Ns, run_timesN):
        # Compute mean and standard deviation of the run times
        mean_time = np.mean(run_time)
        median_time = np.median(run_time)
        std_time = np.std(run_time, ddof=1)

        # Compute the confidence interval
        ci_margin = z_score * (std_time / np.sqrt(len(run_time)))
        lower_ci = mean_time - ci_margin
        upper_ci = mean_time + ci_margin

        # Append results to the lists
        times_mean.append(mean_time)
        times_median.append(median_time)
        times_ci.append((lower_ci, upper_ci))

    # Convert confidence intervals to separate lower and upper bounds for plotting
    lower_ci_bound = [ci[0] for ci in times_ci]
    upper_ci_bound = [ci[1] for ci in times_ci]

    return zip(Ns, times_mean, lower_ci_bound, upper_ci_bound)


def explainer_wrapper(data_x, model, feature_perturbation):
    explainer = shap.TreeExplainer(
        model,
        data_x,
        feature_perturbation=feature_perturbation,
        masker=maskers.Independent(data_x, max_samples=int(1e28))
    ) if feature_perturbation == "interventional" else shap.TreeExplainer(model)

    return explainer(data_x)


def load_model_and_data():
    data = pd.read_csv('data/data.csv')
    # Load the model
    model = xgb.Booster()
    model.load_model('data/model.model')

    return data, model


if __name__ == '__main__':
    data, model = load_model_and_data()
    treeshap_explainer = lambda data: explainer_wrapper(data, model, 'tree_path_dependent')
    background_explainer = lambda data: explainer_wrapper(data, model, 'interventional')

    # benchmark_res_treeshap = benchmark_explainer(treeshap_explainer, data, to_try_N=range(100, 2100, 100))
    benchmark_res_background = benchmark_explainer(background_explainer, data, to_try_N=range(4000, 9000, 1000), num_runs=1)
    # plot_benchmark_times(benchmark_res_background)
    bench_times = get_benchmark_times(benchmark_res_background)
    bench_times_list = list(bench_times)
    df = pd.DataFrame(bench_times_list, columns=['N', 'Mean Time', 'Lower CI', 'Upper CI'])

    # Save the DataFrame to a CSV file
    df.to_csv('/Users/sqf320/Documents/glex/simulation/res/bench_times_true2.csv', index=False)