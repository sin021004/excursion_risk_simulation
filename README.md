# Excursion Risk Simulation

This project simulates excursion risk using both Brownian Motion and Fractional Brownian Motion (fBM). It provides tools to analyze path properties such as delta excursions, local time, truncated variation (upcrossing and downcrossing), and robust estimation of the Hurst parameter.

## Features

- **Brownian Motion and Fractional Brownian Motion classes**
  - Generate random walks and fBM paths with customizable Hurst parameter.
- **Delta Excursion Analysis**
  - Compute and analyze the number and length of excursions above a threshold.
- **Truncated Variation**
  - Calculate upcrossings and downcrossings for path variation analysis.
- **Hurst Parameter Estimation**
  - Estimate the Hurst exponent using both rescaled range analysis and excursion scaling (p = 1/H).
- **Reference Papers**
  - Mathematical background and proofs are provided in the `Reference/` folder.

## Usage

1. Clone the repository and install dependencies (`numpy`, `matplotlib`, `sympy`).
2. Use the provided classes in `Excursion_risk_simulation.py` to generate paths and analyze excursions.
3. Estimate the Hurst parameter using the robust excursion-based method.

## References

See the `Reference/` directory for foundational papers on Brownian motion, fractional Brownian motion, excursion theory, and Hurst estimation.

## Function Overview

### Brownian Class

- `__init__(initial=0)`: Initialize a Brownian motion process.
- `gen_random_walk(step_num)`: Generate a simple symmetric random walk.
- `gen_normal_walk(step_num)`: Generate a random walk with normal increments.
- `calculate_delta_excursions(delta_val)`: Compute intervals and statistics for excursions above a threshold.
- `print_delta_interval()`: Print the list of delta intervals.
- `print_last_delta()`: Print the list of last deltas.
- `print_delta_diff()`: Print the list of delta differences.
- `calculate_truncated_variation(constant_C)`: Compute truncated variation intervals for a given constant.
- `print_truncated_variation()`: Print the list of truncated variation intervals.

### Fractional_Brownian Class

- `__init__(initial=0, Hurst=0.5)`: Initialize a fractional Brownian motion process with a given Hurst parameter.
- `gen_walk(step_num)`: Generate a walk based on the Hurst parameter.
- `calculate_delta_excursions(delta_val)`: Compute intervals and statistics for excursions above a threshold.
- `print_delta_interval()`: Print the list of delta intervals.
- `print_last_delta()`: Print the list of last deltas.
- `print_delta_diff()`: Print the list of delta differences.
- `calculate_truncated_variation(constant_C)`: Compute truncated variation intervals for a given constant.
- `print_truncated_variation()`: Print the list of truncated variation intervals.
- `calculate_P_value(deltas=None)`: Numerically estimate the scaling exponent p as delta approaches 0.
- `calculate_delta_excursion_limit()`: Estimate the limit of delta excursion length as delta approaches 0.
- `print_delta_excursion_limit()`: Print the delta excursion limit.
- `calculate_down_crossing(delta_val)`: Count the number of downcrossings of a threshold.
- `estimate_hurst()`: Estimate the Hurst parameter using rescaled range analysis.
- `count_excursions(delta)`: Count the number of excursions above a threshold.
- `estimate_p_and_hurst_from_excursions(min_exp, max_exp, num)`: Estimate the scaling exponent p and Hurst parameter H from excursion counts.