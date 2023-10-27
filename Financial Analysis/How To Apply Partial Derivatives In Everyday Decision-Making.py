##########################################################################################
# Surface area calculation using partial derivatives
##########################################################################################

import sympy as sp

# Define the variables and the function
L, W, H, A = sp.symbols("L W H A")
surface_area = 2 * L * W + 2 * L * H + 2 * W * H

# Calculate the partial derivatives
partial_derivative_L = sp.diff(surface_area, L)
partial_derivative_W = sp.diff(surface_area, W)
partial_derivative_H = sp.diff(surface_area, H)

# Substitute sample values
values = {L: 3, W: 2, H: 1}

# Calculate the partial derivatives with the sample values
partial_derivative_L_value = partial_derivative_L.subs(values)
partial_derivative_W_value = partial_derivative_W.subs(values)
partial_derivative_H_value = partial_derivative_H.subs(values)

print(f"Partial derivative of A with respect to L: {partial_derivative_L_value}")
print(f"Partial derivative of A with respect to W: {partial_derivative_W_value}")
print(f"Partial derivative of A with respect to H: {partial_derivative_H_value}")


##########################################################################################
# Sensitivity Analysis in Business Profit
##########################################################################################

# Define the variables and the profit function
Q, P, c, F, profit = sp.symbols("Q P c F profit")
profit_formula = Q * (P - c) - F

# Calculate the partial derivatives
partial_derivative_Q = sp.diff(profit_formula, Q)
partial_derivative_P = sp.diff(profit_formula, P)

# Sample values
values = {Q: 100, P: 3.5, c: 2, F: 200}

# Calculate the partial derivatives with the sample values
sensitivity_Q = partial_derivative_Q.subs(values)
sensitivity_P = partial_derivative_P.subs(values)

print(f"Sensitivity of profit to changes in Q: {sensitivity_Q}")
print(f"Sensitivity of profit to changes in P: {sensitivity_P}")


##########################################################################################
# A contour  plot visualization
##########################################################################################


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variables and the profit function
Q, P, c, F, profit = sp.symbols("Q P c F profit")
profit_formula = Q * (P - c) - F

# Sample values
c_value = 2  # 1.2
F_value = 200  # 500

# Create a range of values for Q and P
Q_values = np.linspace(50, 150, 100)  # Quantity of loaves sold
P_values = np.linspace(2, 5, 100)  # Selling price per loaf

# Create an empty matrix to store profit values
profits = np.zeros((len(Q_values), len(P_values)))

# Calculate profits for different combinations of Q and P
for i, q in enumerate(Q_values):
    for j, p in enumerate(P_values):
        values = {Q: q, P: p, c: c_value, F: F_value}
        profits[i, j] = profit_formula.subs(values)

# Calculate the partial derivatives
partial_derivative_Q = sp.diff(profit_formula, Q)
partial_derivative_P = sp.diff(profit_formula, P)

# Calculate sensitivities
sensitivity_Q = partial_derivative_Q.subs({Q: 100, P: 2.5, c: c_value, F: F_value})
sensitivity_P = partial_derivative_P.subs({Q: 100, P: 2.5, c: c_value, F: F_value})

# Create a contour plot
plt.figure(figsize=(12, 6))
contour = plt.contour(P_values, Q_values, profits, levels=20, cmap="viridis")
plt.clabel(contour, inline=1, fontsize=10)
plt.xlabel("Selling Price per Loaf (P)")
plt.ylabel("Quantity of Loaves Sold (Q)")
plt.title("Profit Sensitivity Analysis")
plt.scatter(2.5, 100, color="red", marker="o", label="Current Values")
plt.legend()
plt.colorbar(contour)
plt.grid(True)

plt.show()

print(f"Sensitivity of profit to changes in Q: {sensitivity_Q}")
print(f"Sensitivity of profit to changes in P: {sensitivity_P}")
