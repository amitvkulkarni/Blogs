##########################################################################################################################
# Deterministic Models
##########################################################################################################################


# Parameters
principal = 1000  # Initial investment
annual_rate = 0.05  # Annual interest rate (5%)
years = 10  # Number of years

# Calculate future value using the compound interest formula
future_value = principal * (1 + annual_rate) ** years

print(f"The deterministic future value of the investment is: {future_value:.2f}")


##########################################################################################################################
# Stochastic Models
##########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Parameters
principal = 1000  # Initial investment
annual_rate = 0.05  # Annual interest rate (5%)
years = 10  # Number of years

# Calculate future value for each year
future_values = [principal * (1 + annual_rate) ** year for year in range(years + 1)]

# Plot the deterministic future value as a line plot
plt.plot(range(years + 1), future_values, marker="o", linestyle="-", color="blue")
plt.xlabel("Years")
plt.ylabel("Future Value")
plt.title("Deterministic Future Value Over Time")
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
principal = 1000  # Initial investment
annual_rate_mean = 0.05  # Mean annual interest rate (5%)
annual_rate_stddev = 0.2  # Standard deviation of annual rate (20%)
years = 10  # Number of years
num_simulations = 1000  # Number of Monte Carlo simulations

# Simulate future values using a stochastic model
np.random.seed(0)  # Set a seed for reproducibility
future_values = []

for _ in range(num_simulations):
    # Generate random annual rates with a normal distribution
    annual_rates = np.random.normal(annual_rate_mean, annual_rate_stddev, years)

    # Calculate future value for each year
    future_value_path = [principal]
    for rate in annual_rates:
        future_value_path.append(future_value_path[-1] * (1 + rate))

    future_values.append(future_value_path[-1])

# Plot the distribution of future values
plt.hist(future_values, bins=30, edgecolor="k")
plt.xlabel("Future Value")
plt.ylabel("Frequency")
plt.title("Stochastic Future Value Distribution")
plt.show()
