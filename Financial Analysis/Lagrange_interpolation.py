######################################################################
# Introduction
######################################################################

import numpy as np
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)  # for reproducibility
x = np.random.rand(100)  # 100 random x-values between 0 and 1
y = 3 * x + 2 + np.random.normal(0, 0.2, 100)  # Random y-values with noise

# Perform linear regression to find the best-fit line
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Create a scatter plot of the data points
plt.scatter(x, y, label="Data Points", color="blue")

# Plot the best-fit line
x_range = np.linspace(0, 1, 100)  # Generate a range of x-values for the line
plt.plot(x_range, best_fit_line(x_range), label="Best Fit Line", color="red")

# Add labels and a legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Display the plot
# plt.title('Random Data and Best Fit Line')
plt.show()


######################################################################
# Estimating with linear regression
######################################################################


import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 18, 8, 12, 20])

# Perform linear regression
coefficients = np.polyfit(
    x, y, 1
)  # Fit a linear regression model (1st-degree polynomial)
y_fit = np.polyval(coefficients, x)  # Generate y values for the regression line

# Create a scatter plot of the data points
plt.scatter(x, y, label="Data Points", c="b", marker="o")

# Plot the regression line
plt.plot(x, y_fit, "r", label="Regression Line")

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.title("Scatter Plot with Linear Regression Line")
plt.grid(True)
plt.show()

######################################################################
# Estimating with linear interpolation
######################################################################


import numpy as np
import matplotlib.pyplot as plt

# Sample data for linear interpolation
x = np.array([1, 2, 3, 4, 5])  # Known data points on the x-axis
y = np.array([10, 18, 8, 12, 20])  # Corresponding data points on the y-axis

# Define the point for which you want to estimate the value
x_interpolate = 2.5  # The point between 2 and 3

# Perform linear interpolation
y_interpolate = np.interp(x_interpolate, x, y)

# Plot the data points and the linear interpolation
plt.figure(figsize=(8, 6))
plt.plot(x, y, "o", label="Data Points")
plt.plot(
    x_interpolate, y_interpolate, "ro", label=f"Interpolated Value at x={x_interpolate}"
)
plt.plot(x, y, "r--")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Interpolation")
plt.grid(True)
plt.show()

# Print the interpolated value
print(f"Interpolated value at x={x_interpolate}: {y_interpolate:.2f}")


######################################################################
# Estimating with Lagrange interpolation
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 18, 8, 12, 20])

# Perform Lagrange interpolation
poly = lagrange(x, y)

# Interpolate the value at x = 2.5
x_interp = 2.5
y_interp = poly(x_interp)

# Create a scatter plot of the data points
plt.scatter(x, y, label="Data Points", c="b", marker="o")

# Plot the Lagrange interpolation polynomial
x_range = np.linspace(min(x), max(x), 100)
y_range = poly(x_range)
plt.plot(x_range, y_range, "r", label="Lagrange Interpolation", linestyle="--")

# Highlight the interpolated point
plt.scatter(
    x_interp, y_interp, c="r", marker="o", label=f"Interpolated at x={x_interp}"
)

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.title("Lagrange Interpolation and Interpolated Point")
plt.grid(True)
plt.show()

# Print the interpolated value
print(f"Interpolated value at x = {x_interp}: {y_interp:.2f}")


######################################################################
# Other interpolation technique - Spline interpolation
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 18, 8, 12, 20])

# Perform cubic spline interpolation
cs = CubicSpline(x, y)

# Interpolate the value at x = 2.5
x_interp = 2.5
y_interp = cs(x_interp)

# Create a scatter plot of the data points
plt.scatter(x, y, label="Data Points", c="b", marker="o")

# Plot the cubic spline interpolation
x_range = np.linspace(min(x), max(x), 100)
y_range = cs(x_range)
plt.plot(x_range, y_range, "r", label="Cubic Spline Interpolation", linestyle="--")

# Highlight the interpolated point
plt.scatter(
    x_interp, y_interp, c="r", marker="o", label=f"Interpolated at x={x_interp}"
)

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.title("Cubic Spline Interpolation and Interpolated Point")
plt.grid(True)
plt.show()

# Print the interpolated value
print(f"Interpolated value at x = {x_interp}: {y_interp:.2f}")


######################################################################
# Practical application - Yield curve
######################################################################

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Sample data for the yield curve
maturities = np.array([1, 2, 3, 5, 7, 10])  # Maturities in years
interest_rates = np.array(
    [0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
)  # Corresponding interest rates

# Define the maturity for which you want to estimate the interest rate
target_maturity = 4  # The maturity for which we want to estimate the interest rate

# Create a Lagrange interpolating polynomial
polynomial = lagrange(maturities, interest_rates)

# Estimate the interest rate for the target maturity
estimated_interest_rate = polynomial(target_maturity)

# Plot the yield curve and the estimated interest rate
maturities_interpolated = np.linspace(min(maturities), max(maturities), 100)
interest_rates_interpolated = polynomial(maturities_interpolated)

plt.figure(figsize=(8, 6))
plt.plot(maturities, interest_rates, "o", label="Actual Yield Curve")
plt.plot(
    target_maturity,
    estimated_interest_rate,
    "ro",
    label=f"Estimated Rate at {target_maturity} years",
)
plt.plot(
    maturities_interpolated,
    interest_rates_interpolated,
    label="Interpolated Curve",
    linestyle="--",
)
plt.xlabel("Maturity (Years)")
plt.ylabel("Interest Rate")
plt.legend()
plt.title("Yield Curve Interpolation")
plt.grid(True)
plt.show()

# Print the estimated interest rate
print(
    f"Estimated Interest Rate at {target_maturity} years: {estimated_interest_rate:.4f}"
)
# Print the polynomial equation
poly_equation = np.poly1d(polynomial)
print("Interpolating Polynomial:")
print(poly_equation)


######################################################################
# Practical application - population
######################################################################

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Sample data for the population of a city
years = [2000, 2005, 2010, 2015, 2020]  # Years
population = [
    12000,
    15000,
    18000,
    22000,
    26000,
]  # Corresponding population in the city (in thousands)

# Define the year for which you want to estimate the population
target_year = 2017

# Create a Lagrange interpolating polynomial
polynomial = lagrange(years, population)

# Estimate the population for the target year using Lagrange interpolation
estimated_population = polynomial(target_year)

# Plot the population data and the estimated population
years_interpolated = np.linspace(min(years), max(years), 100)
population_interpolated = polynomial(years_interpolated)

plt.figure(figsize=(8, 6))
plt.plot(years, population, "o", label="Population Data")
plt.plot(
    target_year,
    estimated_population,
    "ro",
    label=f"Estimated Population in {target_year}",
)
plt.plot(
    years_interpolated,
    population_interpolated,
    label="Interpolated Curve",
    linestyle="--",
)
plt.xlabel("Year")
plt.ylabel("Population (thousands)")
plt.legend()
plt.title("Population Growth Interpolation")
plt.grid(True)
plt.show()

# Print the estimated population
print(f"Estimated Population in {target_year}: {estimated_population:.0f} thousands")


#################################################################################
# Additional Analysis (not part of the blog) -
# Comparing Lagrange and Taylor approximations
#################################################################################


import numpy as np
from scipy.interpolate import lagrange
import sympy as sp

# Sample data for the yield curve
maturities = np.array([1, 2, 3, 5, 7, 10])  # Maturities in years
interest_rates = np.array(
    [0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
)  # Corresponding interest rates

# Define the target maturity for which you want to estimate the interest rate
target_maturity = 4  # The maturity for which we want to estimate the interest rate

# Create a Lagrange interpolating polynomial
polynomial = lagrange(maturities, interest_rates)

# Define the polynomial symbolically using sympy
x = sp.Symbol("x")
lagrange_polynomial = sp.Poly(polynomial, x)

# Calculate the estimated interest rate at the target maturity using Lagrange interpolation
estimated_interest_rate = lagrange_polynomial.subs(x, target_maturity)

# Define a small increment for Taylor's approximation
delta_x = 0.5

# Use Taylor's approximation to refine the estimate
taylor_approximation = estimated_interest_rate + delta_x * lagrange_polynomial.diff(
    x
).subs(x, target_maturity)

# Print the estimated interest rate using Lagrange interpolation and Taylor's approximation
print(
    f"Estimated Interest Rate at maturity {target_maturity} years (Lagrange): {estimated_interest_rate:.4f}"
)
print(
    f"Estimated Interest Rate at maturity {target_maturity} years (Taylor's Approximation): {taylor_approximation:.4f}"
)
