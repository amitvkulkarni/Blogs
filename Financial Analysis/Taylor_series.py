from sympy import symbols, diff, pretty_print, factorial, exp
x, y = symbols("x y")
 
n = 5    # Number of iterations
x0 = 0    # The value of "a" or the point
 
func = exp(x)             # The function we are approximating
result = func.subs(x, x0)  # Initializing result with the first term
 
for i in range(1, n):
    result += diff(func, x, i).subs(x, x0) * ((x - x0)**i)/(factorial(i))
    rel_error = (func.subs(x, 3) - result.subs(x, 3))/func.subs(x, 3)
    print("Error: ", float(rel_error))
    print("Value: ", result)
 
pretty_print(result)



#########################################
# Conventional approach
#########################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# plotting exp(x)
def exponential(x):
    return np.exp(x)

x_point = 0.25
x_tangent = np.linspace(x_point - 1, x_point + 1, 100)

highlight_x = [0,0.25, 1]
highlight_y = [np.exp(0), np.exp(0.25), np.exp(1)]
# Create a plot
plt.figure(figsize=(6, 4))
plt.plot(x_tangent, exponential(x_tangent), label=f"Curve of $e^x$", color="blue")
plt.scatter(highlight_x, highlight_y, color="red", label="Points (0, 1)")
# plt.legend()
plt.grid(True)
plt.show()


#########################################
# Linear approximation
#########################################

# plotting exp(x)
def exponential(x):
    return np.exp(x)

x_point = 0
y_point = exponential(x_point)
x_tangent = np.linspace(x_point - 1, x_point + 1, 100)
m = np.exp(x_point)  # The derivative of e^x is e^x itself
b = y_point - m * x_point
# Calculate the corresponding y values for the tangent line
y_tangent = m * x_tangent + b

highlight_x = [0,0.25, 1]
highlight_y = [np.exp(0), np.exp(0.25), np.exp(1)]
# Create a plot
plt.figure(figsize=(6, 4))
plt.plot(x_tangent, exponential(x_tangent), label=f"Curve of $e^x$", color="blue")
plt.plot(x_tangent, y_tangent, label=f"Tangent at x = {x_point}", color="black")
plt.scatter(highlight_x, highlight_y, color="red", label="Points (0, 1)")
# plt.legend()
plt.grid(True)
plt.show()
print(f"Tangent line equation at x = {x_point}: y = {m:.2f}x + {b:.2f}")




#########################################
# Linear and quadratic approximation
#########################################

# x = np.linspace(-2, 2, 400)
# y = np.exp(x)

# Define the function e^x
def exponential(x):
    return np.exp(x)

def quadratic(x):
    return (0.5*(x**2) + x + 1)
    

# Define the point at which you want to draw the tangent
x_point = 0 # You can change this point as needed
y_point = exponential(x_point)
y_point_quad = exponential(x_point)

# Define the tangent line equation at the given point
# The tangent line has the form y = mx + b, where m is the derivative of e^x at the point
m = np.exp(x_point)  # The derivative of e^x is e^x itself
b = y_point - m * x_point

# Define the x values for the tangent line
x_tangent = np.linspace(x_point - 1, x_point + 1, 100)

# Calculate the corresponding y values for the tangent line
y_tangent = m * x_tangent + b

highlight_x = [0,0.25, 1]
highlight_y = [np.exp(0), np.exp(0.25), np.exp(1)]

# Create a plot
plt.figure(figsize=(6, 4))
# plt.plot(x, y, label=f"Curve of $e^x$", color="red")
plt.plot(x_tangent, y_tangent, label=f"Linear Approximation at x = {x_point}", color="black")
plt.plot(x_tangent, exponential(x_tangent), label=f"Curve of $e^x$", color="blue")
plt.plot(x_tangent, quadratic(x_tangent), label="Quadratic Approximation", color="green")
plt.scatter(highlight_x, highlight_y, color="red", label="Points (0, 1)")

# Mark the point of tangency
plt.scatter(x_point, y_point, color="green", label="Point of Tangency")

# Set plot labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Line to $e^x$ at a Specific Point")

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



#########################################
# Quadratic approximation example 2 - exp(2*x+3*x**2)
#########################################

# Define the function e^x
def exponential(x):
    return np.exp(2*x+3*x**2)
    

def quadratic(x):
    #return (0.5*(x**2) + x + 1)
    return (5*(x**2) + 2*x + 1)


x= [-0.3,-0.25,-0.2,-0.1,0,0.1,0.2,0.25,0.3]
pd.DataFrame([[i, round(np.exp(2*i+3*i**2),2),round(quadratic(i),2)] for i in x], columns= ['X', 'Actual','Approximate'])



#########################################
# Taylor series
#########################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Define the function for the Taylor series approximation of e^x
def taylor_approximation(x, n_terms):
    approx = 0
    for n in range(n_terms):
        term = x**n / np.math.factorial(n)
        approx += term
    return round(approx,5)

# Define the range of x values for the plot
x = np.linspace(-1, 1, 100)
x = np.round(x,5)
# x1 = 0.25253

p1 = 0.25253

# Specify the number of terms in the Taylor series expansion
n_terms = 2   # Adjust the number of terms as needed

# Calculate the y values using the Taylor series approximation
# y_taylor = [taylor_approximation(val, n_terms) for val in x]
# y_taylor = [taylor_approximation(val, n_terms) if val==p1 else "" for val in x]
y_taylor = [taylor_approximation(val, n_terms) for val in x if val==p1][0]

# taylor_approximation(p1, n_terms)

# Calculate the actual values of e^x
y_exact = np.exp(p1)

# df= pd.DataFrame(list(zip(y_exact, y_taylor, p1, y_exact - y_taylor)), columns = ["Actual", "Taylor Approx", "iteration", "Error"])
df= pd.DataFrame([[y_exact, y_taylor, p1, y_exact - y_taylor]], columns = ["Actual", "Taylor Approx", "At point", "Error"])
df


##################################################################################
# Application of Taylor series in Finance
##################################################################################


def current_value(face_value, maturity_year, interest_rate, t):
    PV =  face_value * np.exp(-interest_rate * (maturity_year-t))
    return f'The value at {t}ᵗʰ year = {round(PV,4)}'



current_value(100,30,0.05, 12.5)



def taylor_approximation(face_value, interest_rate, maturity_year, t, num_terms):
    # Initialize the approximation
    approximation = 0.0
    
    # Calculate the Taylor series expansion
    for n in range(num_terms):
        term = face_value*(-interest_rate * (maturity_year - t)) ** n / np.math.factorial(n)
        approximation += term    
    return approximation

# Example usage:
face_value = 100  # Face value of the bond
interest_rate = 0.05  # Annual interest rate (5%)
maturity_year = 30  # Maturity year of the bond
t = 12.5  # Time parameter for the approximation

num_terms = 6  # Number of terms in the Taylor series expansion

result = taylor_approximation(face_value, interest_rate, maturity_year, t, num_terms)
print("The approximated value at the specified year = ", round(result,4))

# m = list(range(1,11))
# [[i, taylor_approximation(face_value, interest_rate, maturity_year, t, i)] for i in m]

