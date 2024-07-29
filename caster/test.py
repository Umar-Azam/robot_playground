
# %%

print("hello world")

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%

# Constants
r = 0.1  # wheel radius
l = 0.2  # caster offset
dt = 0.01  # time step
total_time = 10  # total simulation time

# Initial conditions (unstable equilibrium)
x0, y0 = 0, 0
gamma0 = np.pi  # wheel aligned opposite to motion direction
v0 = 0.5  # initial velocity

# Arrays to store simulation data
t = np.arange(0, total_time, dt)
x = np.zeros_like(t)
y = np.zeros_like(t)
gamma = np.zeros_like(t)

# Set initial conditions
x[0], y[0], gamma[0] = x0, y0, gamma0

def update_state(x_prev, y_prev, gamma_prev, v):
    # Compute derivatives
    x_dot = v * np.cos(gamma_prev)
    y_dot = v * np.sin(gamma_prev)
    gamma_dot = (x_dot * np.sin(gamma_prev) - y_dot * np.cos(gamma_prev)) / l
    
    # Update state
    x_new = x_prev + x_dot * dt
    y_new = y_prev + y_dot * dt
    gamma_new = gamma_prev + gamma_dot * dt
    
    return x_new, y_new, gamma_new

# Run simulation
for i in range(1, len(t)):
    x[i], y[i], gamma[i] = update_state(x[i-1], y[i-1], gamma[i-1], v0)
    
    # Add small perturbation to create instability
    if i == len(t) // 10:
        gamma[i] += 0.1

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

def animate(i):
    ax.clear()
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Caster Wheel Simulation (t = {t[i]:.2f}s)')
    
    # Plot trajectory
    ax.plot(x[:i+1], y[:i+1], 'b-')
    
    # Plot wheel
    wheel_x = x[i] + l * np.cos(gamma[i])
    wheel_y = y[i] + l * np.sin(gamma[i])
    ax.plot([x[i], wheel_x], [y[i], wheel_y], 'k-', linewidth=2)
    circle = plt.Circle((wheel_x, wheel_y), r, fill=False)
    ax.add_artist(circle)
    
    # Plot wheel direction
    direction_x = wheel_x + r * np.cos(gamma[i])
    direction_y = wheel_y + r * np.sin(gamma[i])
    ax.plot([wheel_x, direction_x], [wheel_y, direction_y], 'r-', linewidth=1)

ani = FuncAnimation(fig, animate, frames=len(t), interval=50, repeat=False)
plt.show()