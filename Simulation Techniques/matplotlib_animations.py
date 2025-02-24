#####################################################################
# Double pendulum simulation
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Physical constants
g = 9.81  # gravity (m/s²)
L1, L2 = 1.0, 1.0  # lengths of the pendulums
m1, m2 = 1.0, 1.0  # masses of the pendulums
dt = 0.05  # time step

# Initial conditions (angles in radians, angular velocities)
theta1, theta2 = np.pi / 2, np.pi / 2
omega1, omega2 = 0.0, 0.0

# Function to compute acceleration using Lagrangian equations
def derivatives(state):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    den1 = (2 * m1 + m2 - m2 * np.cos(2 * delta))
    den2 = (L2 / L1) * den1

    domega1 = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 
               2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))) / (L1 * den1)

    domega2 = (2 * np.sin(delta) * (omega1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + 
               omega2**2 * L2 * m2 * np.cos(delta))) / (L2 * den2)

    return omega1, domega1, omega2, domega2

# Simulation data
num_frames = 300
states = np.zeros((num_frames, 4))
states[0] = [theta1, omega1, theta2, omega2]

# Runge-Kutta method for solving differential equations
for i in range(1, num_frames):
    k1 = np.array(derivatives(states[i - 1])) * dt
    k2 = np.array(derivatives(states[i - 1] + k1 / 2)) * dt
    k3 = np.array(derivatives(states[i - 1] + k2 / 2)) * dt
    k4 = np.array(derivatives(states[i - 1] + k3)) * dt
    states[i] = states[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Convert angles to x, y coordinates
x1 = L1 * np.sin(states[:, 0])
y1 = -L1 * np.cos(states[:, 0])
x2 = x1 + L2 * np.sin(states[:, 2])
y2 = y1 - L2 * np.cos(states[:, 2])

# Setup figure
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Double Pendulum Motion")

# Lines and markers
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], 'r-', alpha=0.5, lw=1)  # Trajectory trace
trail_x, trail_y = [], []

# Animation update function
def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])

    # Store trail points for visualization
    trail_x.append(x2[frame])
    trail_y.append(y2[frame])
    if len(trail_x) > 50:  # Limit trail length
        trail_x.pop(0)
        trail_y.pop(0)

    trace.set_data(trail_x, trail_y)
    return line, trace

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True)

# **Save the animation as an MP4 (or GIF as fallback)**
try:
    ani.save("double_pendulum.mp4", writer=animation.FFMpegWriter(fps=30))
    print("MP4 saved successfully as 'double_pendulum.mp4'.")
except Exception as e:
    print("FFmpeg not found, saving as GIF instead.")
    ani.save("double_pendulum.gif", writer="pillow")
    print("GIF saved successfully")

plt.show()


    
#####################################################################
# Animated bar charts with Matplotlib
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initial Data - GDP in Trillions (Hypothetical)
countries = ["USA", "China", "Japan", "Germany", "India", "UK", "France"]
gdp_values = np.array([25, 18, 5, 4.5, 3.7, 3.2, 2.9])  # Initial GDP values in Trillions

# Sort initially in descending order
sorted_indices = np.argsort(gdp_values)[::-1]  # Sort in descending order
countries = [countries[i] for i in sorted_indices]
gdp_values = gdp_values[sorted_indices]

num_bars = len(countries)
y_positions = np.arange(num_bars)  # Y positions

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(y_positions, gdp_values, color="royalblue", height=0.6)
ax.set_xlim(0, 30)  # Set X-axis limit for GDP in Trillions
ax.set_yticks(y_positions)
ax.set_yticklabels(countries)
ax.set_xlabel("GDP (Trillions USD)")
ax.set_title("GDP Rankings Over Time")

# Dictionary to track moving y-positions
positions_dict = {name: pos for pos, name in enumerate(countries)}

# Create text labels at the end of each bar
labels = [ax.text(value + 0.5, y, f"{value:.1f}T", va='center', fontsize=12) for value, y in zip(gdp_values, y_positions)]

# Update function for animation
def update(frame):
    global gdp_values, countries, positions_dict
    
    # Randomly change GDP values (simulating growth/shrinkage)
    gdp_values += np.random.uniform(-0.5, 0.5, num_bars)
    gdp_values = np.clip(gdp_values, 2, 30)  # Keep values within range

    # Sort values and get new rankings
    sorted_indices = np.argsort(gdp_values)[::-1]  # Sort descending
    sorted_countries = [countries[i] for i in sorted_indices]
    
    # Smoothly move bars to new positions
    for i, name in enumerate(sorted_countries):
        positions_dict[name] += (i - positions_dict[name]) * 0.2  # Smooth transition
    
    new_y_positions = [positions_dict[name] for name in sorted_countries]
    
    # Update bars and labels
    for bar, new_value, new_y, label in zip(bars, gdp_values[sorted_indices], new_y_positions, labels):
        bar.set_width(new_value)
        bar.set_y(new_y)  # Update Y position smoothly
        
        # Update text labels
        label.set_x(new_value + 0.5)  # Position text slightly beyond bar
        label.set_y(new_y)
        label.set_text(f"{new_value:.1f}T")  # Update value text

    # Update labels and ticks
    ax.set_yticks(new_y_positions)
    ax.set_yticklabels(sorted_countries)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=False)

try:
    ani.save("double_pendulum.mp4", writer=animation.FFMpegWriter(fps=30))
    print("MP4 saved successfully")
except Exception as e:
    print("FFmpeg not found, saving as GIF instead.")
    ani.save("countries_GPD.gif", writer="pillow")
    print("GIF saved successfully")

plt.show()


#####################################################################
# Path Finding
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance

# Generate random points
num_points = 100
np.random.seed(42)  # For consistent results
points = np.random.rand(num_points, 2) * 10  # Scale points in 10x10 space

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Nearest Neighbor Path Animation")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# Scatter plot for points
sc = ax.scatter(points[:, 0], points[:, 1], c="blue", s=50, label="Points")

# Annotate points with coordinates
# annotations = [ax.text(x, y, f"({x:.1f}, {y:.1f})", fontsize=10, ha='right') for x, y in points]

# Initialize lines
lines = []
for _ in range(num_points - 1):
    line, = ax.plot([], [], 'r-', lw=2)  # Create red lines for animation
    lines.append(line)

# Initialize the circle that highlights nearest 5 points
highlight_circle = plt.Circle((0, 0), 0, color='lightblue', alpha=0.3)
ax.add_patch(highlight_circle)

# Nearest Neighbor Pathfinding
visited = set()
current_idx = np.random.randint(num_points)  # Start from a random point
visited.add(current_idx)
order = [current_idx]

while len(visited) < num_points:
    remaining_points = [i for i in range(num_points) if i not in visited]
    nearest_idx = min(remaining_points, key=lambda i: distance.euclidean(points[current_idx], points[i]))
    visited.add(nearest_idx)
    order.append(nearest_idx)
    current_idx = nearest_idx

# Animation update function
def update(frame):
    if frame >= len(order) - 1:
        return lines, highlight_circle  # Stop animation if all points are connected

    i, j = order[frame], order[frame + 1]
    x_values = [points[i, 0], points[j, 0]]
    y_values = [points[i, 1], points[j, 1]]
    lines[frame].set_data(x_values, y_values)  # Draw line between points

    # Calculate distance to the 5 nearest points from the current one
    dists = [distance.euclidean(points[i], points[k]) for k in range(num_points) if k != i]
    sorted_dists = sorted(dists)[:1]  # Get distances of 5 nearest points
    radius = max(sorted_dists) if sorted_dists else 0  # Set circle radius
    
    highlight_circle.set_center((points[i, 0], points[i, 1]))  # Move circle to current point
    highlight_circle.set_radius(radius)  # Update circle size

    return lines, highlight_circle

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(order) - 1, interval=700, blit=False)

# # Save as MP4 (or GIF if ffmpeg is unavailable)
try:
    ani.save("animated_bar_chart.mp4", writer=animation.FFMpegWriter(fps=30))
    print("MP4 saved successfully as 'animated_bar_chart.mp4'.")
except Exception as e:
    print("FFmpeg not found, saving as GIF instead.")
    ani.save("path_finder.gif", writer="pillow")
    print("GIF saved successfully")

plt.legend()
plt.show()


#####################################################################
# Crowd Movement Simulation in Smart Cities
# Wildlife Tracking & Migration Analysis
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree

# Initialize parameters
num_people = 50  # Number of people moving
city_size = (20, 20)  # Grid size (City blocks)
step_size = 0.5  # How much each person moves per step

# Random initial positions
positions = np.random.rand(num_people, 2) * city_size

# Define movement preferences (favor horizontal/vertical movement like roads)
directions = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],  # Straight moves
    [1, 1], [-1, -1], [1, -1], [-1, 1]  # Diagonal moves
])

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, city_size[0])
ax.set_ylim(0, city_size[1])
ax.set_xticks(range(city_size[0] + 1))
ax.set_yticks(range(city_size[1] + 1))
ax.grid(True, linestyle="--", linewidth=0.5)

# People scatter plot
people_dots = ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.7)

# Heatmap data
heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=(city_size[0], city_size[1]))
heatmap_img = ax.imshow(heatmap.T, extent=[0, city_size[0], 0, city_size[1]], origin='lower', alpha=0.8, cmap='Reds')

# Update function for animation
def update(frame):
    global positions
    
    # Move people randomly within city grid
    moves = directions[np.random.randint(0, len(directions), size=num_people)]
    new_positions = positions + (moves * step_size)

    # Keep within bounds
    new_positions = np.clip(new_positions, 0, city_size[0])

    # Update positions
    positions[:] = new_positions

    # Update scatter plot
    people_dots.set_offsets(positions)

    # Update congestion heatmap
    heatmap, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=(city_size[0], city_size[1]))
    heatmap_img.set_data(heatmap.T)

    return people_dots, heatmap_img

# Run animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=500, blit=False)
# # Save as MP4 (or GIF if ffmpeg is unavailable)
try:
    ani.save("animated_bar_chart.mp4", writer=animation.FFMpegWriter(fps=30))
    print("MP4 saved successfully as 'animated_bar_chart.mp4'.")
except Exception as e:
    print("FFmpeg not found, saving as GIF instead.")
    ani.save("tracking.gif", writer="pillow")
    print("GIF saved successfully")
plt.show()


#####################################################################
# Stock Price Movement Simulation
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate time-series data
np.random.seed(42)
num_points = 200  # Number of time points
time = np.arange(0, num_points, 1)
stock_prices = np.cumsum(np.random.randn(num_points) * 2) + 100  # Simulated stock prices

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, num_points)
ax.set_ylim(min(stock_prices) - 5, max(stock_prices) + 5)
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Stock Price ($)")
ax.set_title("Stock Price Movement Over Time")

# Line plot initialization
line, = ax.plot([], [], lw=2, color='blue')
point, = ax.plot([], [], 'ro')  # Red dot for latest price
text = ax.text(num_points * 0.9, max(stock_prices), "", fontsize=12, color='red')

# Initialization function for animation
def init():
    line.set_data([], [])
    point.set_data([], [])
    text.set_text("")
    return line, point, text

# Update function for animation
def update(frame):
    if frame == 0:
        return line, point, text  # Skip the first frame to avoid empty sequences

    x_data = time[:frame].tolist()  # Convert to list to avoid issues
    y_data = stock_prices[:frame].tolist()

    if len(x_data) > 0 and len(y_data) > 0:  # Ensure valid data
        line.set_data(x_data, y_data)
        point.set_data([x_data[-1]], [y_data[-1]])  # Wrap in lists to avoid errors
        text.set_text(f"${y_data[-1]:.2f}")  # Show latest stock price
        text.set_position((x_data[-1], y_data[-1]))

    return line, point, text

# Run animation
ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, interval=100, blit=False)

# # Save as MP4 (or GIF if ffmpeg is unavailable)
try:
    ani.save("animated_bar_chart.mp4", writer=animation.FFMpegWriter(fps=30))
    print("MP4 saved successfully as 'animated_bar_chart.mp4'.")
except Exception as e:
    print("FFmpeg not found, saving as GIF instead.")
    ani.save("stockprice.gif", writer="pillow")
    print("GIF saved successfully")

plt.show()


#####################################################################
# Data science Process Simulation
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.cm as cm  # Import colormap

# Define ML workflow stages and descriptions
stages = [
    "Data Loading", "EDA", "Data Cleaning", "Feature Engineering",
    "Outlier Detection", "Model Training", "Hyperparameter Tuning",
    "Model Evaluation", "Deployment", "Monitoring"
]

descriptions = [
    "Load raw data from multiple sources, including relational databases, cloud storage, CSV files, and APIs. Ensure proper data extraction, handle various formats, and verify data integrity. Automate ingestion workflows to minimize errors and streamline processing. Perform initial inspections to identify inconsistencies, missing values, or formatting issues before further analysis.",
    
    "Explore and visualize data using descriptive statistics, histograms, scatter plots, and box plots to understand distributions and correlations. Use heatmaps to analyze feature relationships and detect multicollinearity. Identify missing values and outliers, ensuring the dataset is clean, structured, and ready for preprocessing, transformation, and feature engineering to enhance predictive performance.",
    
    "Handle missing data using imputation techniques like mean, median, mode, or KNN imputation. Remove duplicate records to maintain dataset integrity. Standardize formats, normalize numerical values, and correct inconsistencies in categorical variables. Ensure data consistency across features to improve model robustness and prevent biases affecting machine learning performance.",
    
    "Create new features or transform existing ones to improve model performance by engineering meaningful attributes. Generate polynomial features, interaction terms, and time-based patterns. Apply transformations like log scaling, one-hot encoding, or feature binning. Use dimensionality reduction techniques such as PCA or feature selection to enhance interpretability and prevent overfitting.",
    
    "Detect and handle anomalies or extreme values that may distort model accuracy using statistical methods like Z-score, IQR, or robust clustering techniques such as DBSCAN. Visualize outliers with box plots or scatter plots. Decide whether to remove, transform, or cap extreme values to maintain a balanced dataset and ensure reliable predictions.",
    
    "Train machine learning models on the prepared dataset using various algorithms suited for classification, regression, or clustering tasks. Select models such as Decision Trees, Random Forest, Gradient Boosting, or Neural Networks. Split data into training and validation sets, apply cross-validation, and fit models to capture underlying patterns for accurate predictions.",
    
    "Optimize model performance by fine-tuning hyperparameters using Grid Search, Random Search, or Bayesian Optimization. Experiment with different learning rates, tree depths, regularization techniques, and ensemble methods. Apply cross-validation to validate results and select the best hyperparameter configurations to achieve an optimal balance between model accuracy and generalization.",
    
    "Evaluate model performance using appropriate metrics such as accuracy, precision, recall, F1-score for classification tasks, and RMSE or R² for regression models. Generate confusion matrices, ROC curves, and precision-recall curves for deeper insights. Compare model performance across different approaches and refine models based on data-driven evaluation metrics.",
    
    "Deploy the trained model into a production environment for real-world applications using APIs, microservices, or cloud-based platforms. Ensure scalability and efficiency using containerization with Docker or deployment tools like Flask, FastAPI, or Kubernetes. Monitor inference latency, optimize response times, and implement secure handling of user data for seamless operations.",
    
    "Continuously monitor model performance and retrain as needed to maintain accuracy over time. Track model drift, changes in data distribution, and performance degradation using monitoring tools. Implement automated retraining pipelines and A/B testing to improve predictions. Maintain model versioning and refine deployments to ensure consistent accuracy in evolving environments."
]


num_stages = len(stages)

# 🎨 Choose colormap (Modify this to change colors)
colormap = cm.get_cmap("Pastel1", num_stages)  # Options: 'viridis', 'plasma', 'Blues', 'coolwarm'

# Set up figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")  # Hide axes

# Position variables
center_x, center_y = 0.4, 0.6  # Center position for stages
corner_x, corner_y = 0.15, 0.85  # Top-left position for stacked stages
desc_x, desc_y = 0.35, 0.85  # Position for descriptions
y_spacing = 0.08  # Space between stacked elements

# Function to draw a text box with gradient color
def draw_box(ax, x, y, text, color):
    text_width = 0.18  # Adjusted to fit text
    text_height = 0.05
    box = patches.Rectangle((x - text_width / 2, y - text_height / 2), text_width, text_height,
                            edgecolor="black", facecolor=color, lw=2)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=10, weight="bold", color="black", ha="center", va="center")

# Function to update animation frame
def update(frame):
    ax.clear()  # Clear previous frame
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Draw stored steps in the top-left corner
    for i in range(frame):
        y_offset = corner_y - (i * y_spacing)
        draw_box(ax, corner_x, y_offset, stages[i], colormap(i / num_stages))

    # Show current step in the center moving towards the stack
    if frame < num_stages:
        transition_x = np.linspace(center_x, corner_x, 10)
        transition_y = np.linspace(center_y, corner_y - (frame * y_spacing), 10)

        for i in range(len(transition_x)):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # Draw stored steps
            for j in range(frame):
                y_offset = corner_y - (j * y_spacing)
                draw_box(ax, corner_x, y_offset, stages[j], colormap(j / num_stages))

            # Draw current moving step
            draw_box(ax, transition_x[i], transition_y[i], stages[frame], colormap(frame / num_stages))

            # Show description for the current stage on the right
            ax.text(desc_x, desc_y, descriptions[frame], fontsize=10, color="black",
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"),
                    ha="left", va="center", wrap=True)

            plt.pause(0.08)  # Smooth transition delay

    # Final frame: Show all descriptions next to each stage
    if frame == num_stages:
        for i in range(num_stages):
            y_offset = corner_y - (i * y_spacing)
            draw_box(ax, corner_x, y_offset, stages[i], colormap(i / num_stages))

            # Display all descriptions next to their respective stages
            ax.text(desc_x, y_offset, descriptions[i], fontsize=9, color="black",
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"),
                    ha="left", va="center", wrap=True)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=num_stages + 100, interval=500, repeat=False)

plt.show()


