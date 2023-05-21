import matplotlib.pyplot as plt
import numpy as np

# Define the equations of motion for the hour and minute hand
def dtheta_dt(theta, t):
    return 2*np.pi/12/60  # Hour hand rotates at 1/12 the speed of the minute hand

def dphi_dt(phi, t):
    return 2*np.pi/60  # Minute hand rotates at constant rate

# Define the initial angles of the hands
theta0 = 0
phi0 = 0

# Integrate the equations of motion to find the positions of the hands as a function of time
t = np.linspace(0, 2*60, 2000)  # Time range from 0 to 12 hours
theta = np.zeros(len(t))
phi = np.zeros(len(t))
theta[0] = theta0
phi[0] = phi0

for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    theta[i] = theta[i-1] + dtheta_dt(theta[i-1], t[i-1])*dt
    phi[i] = phi[i-1] + dphi_dt(phi[i-1], t[i-1])*dt

# Calculate the distance between the hands
R = 3  # Length of hour hand
r = 4  # Length of minute hand
xh = R*np.cos(theta)
yh = R*np.sin(theta)
xm = r*np.cos(phi)
ym = r*np.sin(phi)
d = np.sqrt((xh - xm)**2 + (yh - ym)**2)

# Find the time when the distance is increasing most rapidly
d_dt = np.diff(d) / np.diff(t)
max_index = np.argmax(d_dt)
max_time = t[max_index]
distance_at_max_time = d[max_index]

print("The distance between the hands is increasing most rapidly at t =", max_time, "minutes.")
print("The distance between the hands is d =", distance_at_max_time, "units when distance between the hands is increasing most rapidly.")


# Create a subplot showing the solutions of the differential equations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
fig.subplots_adjust(hspace=0.5)
ax1.plot(xh, yh, label='Hour Hand')
ax1.plot(xm, ym, label='Minute Hand')
ax1.set_xlabel('X position')
ax1.set_ylabel('Y position')
ax1.legend()
ax1.set_title('Position of Hour and Minute Hands')

# Plot the rate of change of the distance between the hands
ax2.plot(t[:-1], d_dt)
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('d(distance)/dt')
ax2.set_title('Rate of Change of Distance between Hands')

# Plot the distance between the hands as a function of time
ax3.plot(t, d)
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Distance')
ax3.set_title('Distance between Hands')

plt.show()

