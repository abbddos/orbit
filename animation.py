
import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate as Propagator
from pyquaternion import Quaternion
from astropy import units as u

import matplotlib
matplotlib.use('Qt5Agg') # Or 'Qt5Agg', 'Qt4Agg', 'WXAgg' depending on what's installed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from parameters import Orbit_Parameters

class SpacecraftAnimator:
    def __init__(self, time_data, q_data, r_eci_data, animation_dt_interval_s, full_orbit_path_eci, axis_length_m):
        self.time_history = time_data                 # Time points for animation frames
        self.q_history = q_data                       # Quaternion data for each frame
        self.r_eci_history = r_eci_data               # Spacecraft ECI position for each frame
        self.animation_dt_interval = animation_dt_interval_s # Time interval between animation frames
        self.full_orbit_path_eci = full_orbit_path_eci # Full orbital path for static display
        self.axis_length = axis_length_m              # Length of spacecraft body axes in the plot

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self._setup_plot()

    def _setup_plot(self):
        # Plot Earth Center (using user's specified style)
        self.ax.plot([0], [0], [0], 'bo', markersize=15, label='Earth Center') 
        
        # Plot the full orbital path (static background, using user's specified style)
        self.ax.plot(self.full_orbit_path_eci[:, 0], self.full_orbit_path_eci[:, 1], self.full_orbit_path_eci[:, 2], 
                     'c--', linewidth=1, alpha=0.5, label='Full Orbital Path') 

        # Initialize elements that will be animated: Spacecraft body axes (quiver arrows)
        # Initialize with placeholder values, these will be updated in _update_animation
        self.sc_quiver = self.ax.quiver(0, 0, 0, 0, 0, 0, 
                                        colors=['r', 'g', 'b'], # Red for X, Green for Y, Blue for Z
                                        length=self.axis_length, # Use the passed axis_length
                                        normalize=True)

        # Set plot limits
        max_range = np.max(np.abs(self.full_orbit_path_eci)) * 1.5 # Increased buffer for better view
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Spacecraft Attitude & Orbit Animation')
        self.ax.legend()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.view_init(elev=30, azim=45) # Initial viewing angle

    def _update_animation(self, frame):
        """Updates the plot elements for each animation frame."""
        r_sc = self.r_eci_history[frame] # Spacecraft ECI position for current frame
        q_sc = Quaternion(self.q_history[frame]) # Spacecraft attitude quaternion for current frame

        # Calculate ECI directions of spacecraft body axes
        Bx_dir_eci = q_sc.inverse.rotate(np.array([1, 0, 0])) # Body X-axis in ECI
        By_dir_eci = q_sc.inverse.rotate(np.array([0, 1, 0])) # Body Y-axis in ECI
        Bz_dir_eci = q_sc.inverse.rotate(np.array([0, 0, 1])) # Body Z-axis in ECI

        # Set quiver origins (all at spacecraft current position)
        X_starts = np.array([r_sc[0], r_sc[0], r_sc[0]])
        Y_starts = np.array([r_sc[1], r_sc[1], r_sc[1]])
        Z_starts = np.array([r_sc[2], r_sc[2], r_sc[2]])

        # Set quiver directions (the ECI components of the body axes)
        U_dirs = np.array([Bx_dir_eci[0], By_dir_eci[0], Bz_dir_eci[0]])
        V_dirs = np.array([Bx_dir_eci[1], By_dir_eci[1], Bz_dir_eci[1]])
        W_dirs = np.array([Bx_dir_eci[2], By_dir_eci[2], Bz_dir_eci[2]])

        # Update the quiver plot (axes of the spacecraft)
        self.sc_quiver.set_UVC(U_dirs, V_dirs, W_dirs)
        self.sc_quiver.set_XYZ(X_starts, Y_starts, Z_starts)

        # Return updated plot elements for blitting
        return (self.sc_quiver,)

    def animate(self):
        """Starts the animation."""
        num_frames = len(self.time_history)
        ani = FuncAnimation(self.fig, self._update_animation, frames=num_frames, 
                            interval=self.animation_dt_interval * 1000,
                            blit=True) # Use blit for smoother animation

        plt.show() # Display the animation window