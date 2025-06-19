import numpy as np
import matplotlib

matplotlib.use('WXAgg') # Or 'Qt5Agg', 'Qt4Agg', 'WXAgg' depending on what's installed
from poliastro.bodies import Earth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pyquaternion import Quaternion
from orbit import Orbit_Environment, SpaceCraft 
from parameters import Orbit_Parameters 
from astropy import units as u
from animation import SpacecraftAnimator


# --- Functions for Static Plots ---
def plot_quaternion_history(time_history, q_history):
    """Plots the quaternion components over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, q_history[:, 0], label='$q_w$')
    plt.plot(time_history, q_history[:, 1], label='$q_x$')
    plt.plot(time_history, q_history[:, 2], label='$q_y$')
    plt.plot(time_history, q_history[:, 3], label='$q_z$')
    plt.title('Quaternion Components Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Component')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_angular_velocity_history(time_history, w_history):
    """Plots the angular velocity components over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, w_history[:, 0], label='omega_x (rad/s)')
    plt.plot(time_history, w_history[:, 1], label='omega_y (rad/s)')
    plt.plot(time_history, w_history[:, 2], label='omega_z (rad/s)')
    plt.title('Angular Velocity Components Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_static_orbit(r_eci_history, plot_range_km=None):
    """Plots the static 3D orbital trajectory with Earth."""
    fig_static = plt.figure(figsize=(10, 8))
    ax_static = fig_static.add_subplot(111, projection='3d')

    # Earth sphere
    earth_radius = Earth.R.to_value(u.m)
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = earth_radius * np.cos(u_sphere) * np.sin(v_sphere)
    y_earth = earth_radius * np.sin(u_sphere) * np.sin(v_sphere)
    z_earth = earth_radius * np.cos(v_sphere)
    ax_static.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)

    # Spacecraft orbit path
    ax_static.plot(r_eci_history[:, 0], r_eci_history[:, 1], r_eci_history[:, 2], 
                   'k-', linewidth=1.5, label='Spacecraft Orbit')
    
    # Starting and ending points
    ax_static.plot([r_eci_history[0,0]], [r_eci_history[0,1]], [r_eci_history[0,2]], 
                   'go', markersize=8, label='Start Position')
    ax_static.plot([r_eci_history[-1,0]], [r_eci_history[-1,1]], [r_eci_history[-1,2]], 
                   'rx', markersize=8, label='End Position')

    # Set plot limits
    if plot_range_km is None:
        plot_range_m = Orbit_Parameters.SEMI_MAJOR_AXIS * 1.2 * 1000
    else:
        plot_range_m = plot_range_km * 1000

    ax_static.set_xlim([-plot_range_m, plot_range_m])
    ax_static.set_ylim([-plot_range_m, plot_range_m])
    ax_static.set_zlim([-plot_range_m, plot_range_m])
    ax_static.set_xlabel('X (m)')
    ax_static.set_ylabel('Y (m)')
    ax_static.set_zlabel('Z (m)')
    ax_static.set_title('Spacecraft Orbital Trajectory (ECI)')
    ax_static.legend()
    ax_static.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal for true 3D representation


orbit_env = Orbit_Environment(
        a=Orbit_Parameters.SEMI_MAJOR_AXIS,
        ecc=Orbit_Parameters.ECCENTRICITY,
        inc=Orbit_Parameters.INCLINATION,
        raan=Orbit_Parameters.RAAN,
        argp=Orbit_Parameters.ARGP,
        nu=Orbit_Parameters.TRUE_ANOMALY,
        Cd=Orbit_Parameters.DRAG,
        A=Orbit_Parameters.REFERENCE_AREA,
        rho=Orbit_Parameters.ATMOSPHERIC_DENSITY,
        noise_mean=Orbit_Parameters.NOISE_MEAN,
        noise_std=Orbit_Parameters.NOISE_STD
    )

spacecraft = SpaceCraft(
        q_axis=Orbit_Parameters.INITIAL_ATTITUDE,
        q_angle=Orbit_Parameters.INITIAL_ANGLE,
        shape_vector=Orbit_Parameters.SHAPE_VECTOR,
        inertial_matrix=Orbit_Parameters.INERTIAL_MATRIX,
        initial_linear_velocity=Orbit_Parameters.INITIAL_LINEAR_VELOCITY,
        initial_angular_velocity=Orbit_Parameters.INITIAL_ANGULAR_VELOCITY,
        env=orbit_env # Pass the initialized orbit_env
    )

# --- Simulation Setup ---
# For animation, let's keep the duration shorter initially, e.g., 0.25 of an orbit
simulation_duration_s = orbit_env.time_of_flight.to_value(u.s) * 0.25 
dt_simulation_step = 10.0 # Time step for ODE solver propagation
animation_dt_interval = 2.0 # Time step for animation frames (smaller for smoother animation)

# Initial state
current_q = spacecraft.Initial_Attitude
current_w = spacecraft.w_initial_body
y_initial_state = np.concatenate([current_q.elements, current_w])

# Data storage for animation (finer resolution)
time_history_anim = []
q_history_anim = [] 
w_history_anim = []
r_eci_history_anim = [] 
    
# Store initial state for animation
initial_r_eci, _ = spacecraft.get_orbit_state_at_time(0.0)
time_history_anim.append(0.0)
q_history_anim.append(current_q.elements)
w_history_anim.append(current_w)
r_eci_history_anim.append(initial_r_eci)


# --- Simulation Loop to generate high-resolution data for animation ---
num_major_steps = int(simulation_duration_s / dt_simulation_step)
    
for i in range(num_major_steps):
    t_start = i * dt_simulation_step
    t_end = (i + 1) * dt_simulation_step
    if t_end > simulation_duration_s: 
        t_end = simulation_duration_s

    control_torque = np.zeros(3) # No controller applied

    sol = spacecraft.solver((t_start, t_end), y_initial_state, control_torque)

    if sol.success:
        fine_times_in_step = np.arange(t_start + animation_dt_interval, t_end + 1e-9, animation_dt_interval)
        if not np.isclose(fine_times_in_step[-1], t_end):
            fine_times_in_step = np.append(fine_times_in_step, t_end)
            
        for t_fine in fine_times_in_step:
            y_interp = sol.sol(t_fine)
            q_interp = Quaternion(y_interp[0:4]).unit 
            w_interp = y_interp[4:7]
            r_eci_interp, _ = spacecraft.get_orbit_state_at_time(t_fine)

            time_history_anim.append(t_fine)
            q_history_anim.append(q_interp.elements)
            w_history_anim.append(w_interp)
            r_eci_history_anim.append(r_eci_interp)

        y_initial_state = sol.y[:, -1]
        current_q = Quaternion(y_initial_state[0:4]).unit
        current_w = y_initial_state[4:7]
            
        if i % (num_major_steps // 10 + 1) == 0:
            print(f"Data collection progress: {t_end:.2f}s / {simulation_duration_s:.2f}s")

    else:
        print(f"Simulation terminated early due to ODE solver failure at t={t_end:.2f}s: {sol.message}")
        break
    
print("--- Simulation Complete ---")

print("--- Data Collection Complete. Preparing Animation ---")

# Convert lists to NumPy arrays for easier indexing
time_history_anim = np.array(time_history_anim)
q_history_anim = np.array(q_history_anim)
w_history_anim = np.array(w_history_anim)
r_eci_history_anim = np.array(r_eci_history_anim)
r_eci_full_orbit = orbit_env.get_full_orbit_positions()


r_sc = r_eci_history_anim[-1]
q_sc = Quaternion(q_history_anim[-1])
Bx_dir_eci = q_sc.inverse.rotate(np.array([1, 0, 0]))
By_dir_eci = q_sc.inverse.rotate(np.array([0, 1, 0]))
Bz_dir_eci = q_sc.inverse.rotate(np.array([0, 0, 1]))

# --- Generate Static Plots ---
plot_quaternion_history(time_history_anim, q_history_anim)
plot_angular_velocity_history(time_history_anim, w_history_anim)
# Show all static plots at once
#plt.show()

# --- Create and Run Animation using the new class ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Earth Center (static plot element)
ax.plot([0], [0], [0], 'bo', markersize=15, label='Earth Center')

# Orbital Path (static plot element - shows the full trajectory)
ax.plot(r_eci_full_orbit[:, 0], r_eci_full_orbit[:, 1], r_eci_full_orbit[:, 2],
        'c--', linewidth=1, label='Orbital Path')

axis_length = 1.5e6 # Length of spacecraft axes in meters (from your code)
sc_quiver_animated = ax.quiver(
    [0, 0, 0], [0, 0, 0], [0, 0, 0],  # Initial origins (dummy)
    [1, 0, 0], [0, 1, 0], [0, 0, 1],  # Initial directions (dummy - unit vectors for X, Y, Z)
    colors=['r', 'g', 'b'],           # Colors for X, Y, Z axes
    length=axis_length,               # Length of the arrows
    normalize=True,                   # Directions are normalized unit vectors
    animated=True                     # Mark this artist for blitting
)

# Set plot limits (based on the full orbital path)
max_range = np.max(np.abs(r_eci_history_anim)) * 1.1 # Add a 10% buffer
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Earth Center and Orbital Path with Animated Spacecraft Axes') # Slightly modified title for clarity
ax.legend()
ax.set_aspect('equal', adjustable='box')
ax.view_init(elev=30, azim=45) # Set a good initial viewing angle

# --- ANIMATION UPDATE FUNCTION ---#
def update(frame):
    """
    Updates the spacecraft's quivers (body axes) for each animation frame.
    This function is called by FuncAnimation for each frame.
    """
    r_sc = r_eci_history_anim[frame] # Position vector [x, y, z]
    q_sc = Quaternion(q_history_anim[frame]) # Attitude quaternion [w, x, y, z]

    Bx_dir_eci = q_sc.inverse.rotate(np.array([1, 0, 0])) # Body X-axis (red)
    By_dir_eci = q_sc.inverse.rotate(np.array([0, 1, 0])) # Body Y-axis (green)
    Bz_dir_eci = q_sc.inverse.rotate(np.array([0, 0, 1])) # Body Z-axis (blue)

    segments = [
        [[r_sc[0], r_sc[1], r_sc[2]], [r_sc[0] + Bx_dir_eci[0] * axis_length, r_sc[1] + Bx_dir_eci[1] * axis_length, r_sc[2] + Bx_dir_eci[2] * axis_length]],
        [[r_sc[0], r_sc[1], r_sc[2]], [r_sc[0] + By_dir_eci[0] * axis_length, r_sc[1] + By_dir_eci[1] * axis_length, r_sc[2] + By_dir_eci[2] * axis_length]],
        [[r_sc[0], r_sc[1], r_sc[2]], [r_sc[0] + Bz_dir_eci[0] * axis_length, r_sc[1] + Bz_dir_eci[1] * axis_length, r_sc[2] + Bz_dir_eci[2] * axis_length]]
    ]

    sc_quiver_animated.set_segments(segments)
    sc_quiver_animated.set_color(['r', 'g', 'b'])

    return sc_quiver_animated,

ani = FuncAnimation(fig, update, frames=len(r_eci_history_anim),
                    interval=100, blit=False, repeat=True)

# Show the animation plot window
plt.show()

print("\nAnimation display attempted. Please replace the DUMMY DATA section with your actual simulation outputs for r_eci_history_anim and q_history_anim.")