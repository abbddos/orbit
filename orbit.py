import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate as Propagator
from pyquaternion import Quaternion
from astropy import units as u

class Orbit_Environment:
    def __init__(self, a, ecc, inc, raan, argp, nu, Cd, A, rho, noise_mean, noise_std):
        # Orbital parameters
        self.a = a * u.km  # Semi-major axis
        self.ecc = ecc * u.one  # Eccentricity
        self.inc = inc * u.deg  # Inclination
        self.raan = raan * u.deg  # Right Ascension of Ascending Node
        self.argp = argp * u.deg  # Argument of Perigee
        self.nu = nu * u.deg  # True Anomaly

        # Atmospheric and Drag parameters
        self.Drag = Cd  # Drag coefficient
        self.Reference_Area = A  # Reference area for drag
        self.Atmospheric_Density = rho  # Atmospheric density (kg/m^3)

        # Noise parameters
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        # Initialize the orbit using poliastro
        self.initial_orbit = Orbit.from_classical(Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)

        # Calculate orbital period for time of flight (useful for full orbit simulation)
        self.time_of_flight = self.initial_orbit.period

    def Propagate(self, time_of_flight=None):
        """
        Creates a poliastro propagator instance for the initial orbit.
        If time_of_flight is not provided, uses the orbit's period.
        """
        if time_of_flight is None:
            time_of_flight = self.time_of_flight
        else:
            time_of_flight = time_of_flight * u.s # Assume input is in seconds

        # For propagation, we'll return a function that can propagate from the initial orbit
        # poliastro's propagate function directly takes time from initial orbit
        return lambda time_s: self.initial_orbit.propagate(time_s * u.s)

    def Positions(self, times):
        """
        Calculates spacecraft positions in ECI frame for an array of times.
        Args:
            times (np.ndarray): Array of time values in seconds.
        Returns:
            np.ndarray: Array of position vectors (x, y, z) in meters.
        """
        propagator_func = self.Propagate()
        positions = []
        for t_s in times:
            orbit = propagator_func(t_s)
            positions.append(orbit.r.to_value(u.m))
        return np.array(positions)

    def Nadir_Pointing(self, times):
        """
        Calculates the Nadir pointing vector (negative position vector) in ECI frame
        for an array of times.
        Args:
            times (np.ndarray): Array of time values in seconds.
        Returns:
            np.ndarray: Array of nadir pointing vectors (x, y, z) in meters.
        """
        positions = self.Positions(times)
        nadir_vectors = -positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]
        return nadir_vectors

    def Desired_Body_Quaternions_LVLH(self):
        """
        Generates a trajectory of desired spacecraft body quaternions (ECI to Body)
        for nadir pointing, following the Local Vertical Local Horizontal (LVLH) convention.

        LVLH Frame Definition:
        +Z_body points towards Nadir (anti-radial, towards Earth center).
        +Y_body points along the Orbital Normal (perpendicular to orbital plane).
        +X_body completes the right-handed triad (roughly along orbital velocity).

        Returns:
            A NumPy array of pyquaternion.Quaternion objects representing the desired
            ECI-to-Body attitude for each time step.
        """
        times = np.linspace(0, self.time_of_flight.to_value(u.s), 100) # Use the time_of_flight in seconds
        propagator = self.Propagate() # Get the propagator instance
        desired_body_quaternions = []

        for time_s in times:
            # Get Orbital State Vectors in ECI
            orbit = propagator(time_s)
            r_eci = orbit.r.to_value(u.m) # Position vector in ECI (meters)
            v_eci = orbit.v.to_value(u.m / u.s) # Velocity vector in ECI (m/s)

            # Robustness: Handle very small magnitudes (e.g., at Earth center or zero velocity)
            if np.linalg.norm(r_eci) < 1e-9:
                desired_body_quaternions.append(Quaternion(1,0,0,0)) # Identity quaternion
                continue
            
            # Calculate orbital angular momentum for orbital normal vector
            orbital_angular_momentum_eci = np.cross(r_eci, v_eci)
            if np.linalg.norm(orbital_angular_momentum_eci) < 1e-9:
                desired_body_quaternions.append(Quaternion(1,0,0,0)) # Identity quaternion
                continue

            # Define Desired Body Frame Axes in ECI (LVLH)
            desired_Bz_eci = -r_eci / np.linalg.norm(r_eci) # Z-body points to Nadir
            
            # Y-body points along Orbital Normal
            orbit_normal_eci = orbital_angular_momentum_eci / np.linalg.norm(orbital_angular_momentum_eci)
            desired_By_eci = orbit_normal_eci 

            # X-body completes the right-handed triad (X = Y x Z)
            desired_Bx_eci = np.cross(desired_By_eci, desired_Bz_eci)
            desired_Bx_eci = desired_Bx_eci / np.linalg.norm(desired_Bx_eci)

            # Re-orthogonalize By if needed (due to floating point inaccuracies)
            desired_By_eci = np.cross(desired_Bz_eci, desired_Bx_eci)
            desired_By_eci = desired_By_eci / np.linalg.norm(desired_By_eci)

            # Construct the Rotation Matrix from ECI to Body
            # Rows are Bx_eci, By_eci, Bz_eci (representing ECI axes in body frame)
            # OR columns are ECI axes expressed in body frame coordinates
            # For Quaternion(matrix=R_ECI_to_BODY), R_ECI_to_BODY is a 3x3 matrix whose columns are the ECI basis vectors expressed in the BODY frame
            # or equivalently, whose rows are the BODY basis vectors expressed in the ECI frame.
            R_eci_to_body_desired = np.array([
                desired_Bx_eci,
                desired_By_eci,
                desired_Bz_eci
            ])

            # Correct way to initialize pyquaternion.Quaternion from a rotation matrix
            q_target_eci_to_body = Quaternion(matrix=R_eci_to_body_desired)
            
            desired_body_quaternions.append(q_target_eci_to_body)

        return np.array(desired_body_quaternions)

    def Desired_Body_Angular_Velocities_LVLH(self):
        """
        Generates a trajectory of desired spacecraft body angular velocities
        for nadir pointing (LVLH frame).
        This is typically the orbital angular velocity expressed in the LVLH body frame.

        Returns:
            A NumPy array of 3-element angular velocity vectors (omega_x, omega_y, omega_z)
            in rad/s, expressed in the LVLH body frame.
        """
        times = np.linspace(0, self.time_of_flight.to_value(u.s), 100)
        propagator = self.Propagate()
        desired_body_angular_velocities = []

        for time_s in times:
            orbit = propagator(time_s)
            r_eci = orbit.r.to_value(u.m)
            v_eci = orbit.v.to_value(u.m / u.s)

            if np.linalg.norm(r_eci) < 1e-9 or np.linalg.norm(np.cross(r_eci, v_eci)) < 1e-9:
                desired_body_angular_velocities.append(np.array([0.0, 0.0, 0.0]))
                continue

            # 1. Calculate orbital angular velocity in ECI frame
            # The orbital angular velocity vector (omega_orb_eci) is along the orbital normal
            # Magnitude is ||h|| / ||r||^2, direction is h_unit
            orbital_angular_momentum_eci = np.cross(r_eci, v_eci)
            omega_orbital_eci = orbital_angular_momentum_eci / (np.linalg.norm(r_eci)**2)


            # 2. Get the current desired body frame (LVLH) rotation matrix from ECI
            # This needs to be consistent with how the quaternion is formed
            desired_Bz_eci = -r_eci / np.linalg.norm(r_eci)
            orbit_normal_eci = orbital_angular_momentum_eci / np.linalg.norm(orbital_angular_momentum_eci)
            desired_By_eci = orbit_normal_eci
            desired_Bx_eci = np.cross(desired_By_eci, desired_Bz_eci)
            desired_Bx_eci = desired_Bx_eci / np.linalg.norm(desired_Bx_eci)
            desired_By_eci = np.cross(desired_Bz_eci, desired_Bx_eci) # Re-orthogonalize
            desired_By_eci = desired_By_eci / np.linalg.norm(desired_By_eci)

            # R_eci_to_body_current transforms a vector from ECI to Body
            R_eci_to_body_current = np.array([
                desired_Bx_eci,
                desired_By_eci,
                desired_Bz_eci
            ])

            # 3. Transform orbital angular velocity from ECI to desired Body frame
            omega_target_body = R_eci_to_body_current @ omega_orbital_eci

            desired_body_angular_velocities.append(omega_target_body)

        return np.array(desired_body_angular_velocities)





class SpaceCraft:
    def __init__(self, q_axis, q_angle, shape_vector, inertial_matrix, initial_linear_velocity, initial_angular_velocity, env):
        # Initial Attitude (Quaternion representing ECI to Body rotation)
        self.Initial_Attitude = Quaternion(axis=q_axis, angle=q_angle) # Q_ECI_to_Body

        # Spacecraft geometric and inertial properties
        self.Shape_Vector = shape_vector # Defined in Body Frame (e.g., center of pressure offset)
        self.Inertial_Matrix = inertial_matrix # Defined in Body Frame (3x3 matrix)

        # Initial Linear and Angular Velocities
        # initial_linear_velocity is typically in ECI, used for orbital state in propagation
        self.v_initial_eci = initial_linear_velocity 
        # initial_angular_velocity is in Body frame (rad/s)
        self.w_initial_body = initial_angular_velocity 

        # Reference to the Orbit_Environment instance
        self.env = env

    def get_orbit_state_at_time(self, time_s):
        """
        Gets the current orbital position (r_eci) and velocity (v_eci) in ECI frame
        by propagating the initial orbit to the given time.
        Args:
            time_s (float): Time in seconds from the initial epoch.
        Returns:
            tuple: (r_eci (np.ndarray), v_eci (np.ndarray)) in meters and m/s respectively.
        """
        # Ensure the propagator function from env.Propagate() is used correctly
        propagator_func = self.env.Propagate()
        orbit = propagator_func(time_s)
        r_eci = orbit.r.to_value(u.m) # Position in ECI (meters)
        v_eci = orbit.v.to_value(u.m / u.s) # Velocity in ECI (m/s)
        return r_eci, v_eci

    def calculate_aerodynamic_torque_body(self, current_q_eci_to_body: Quaternion, current_v_eci: np.ndarray):
        """
        Calculates aerodynamic torque and expresses it in the Body Frame.
        Args:
            current_q_eci_to_body (Quaternion): Current attitude of Body relative to ECI.
            current_v_eci (np.ndarray): Current linear velocity of spacecraft in ECI.
        Returns:
            np.ndarray: Aerodynamic torque vector in the Body Frame (N*m).
        """
        # Transform ECI velocity to Body frame
        # current_q_eci_to_body is Q_ECI_to_Body. To rotate a vector FROM ECI TO Body, use .rotate()
        v_body = current_q_eci_to_body.rotate(current_v_eci)

        # Calculate drag force in Body frame
        # Assuming drag acts opposite to velocity vector
        F_drag_body = -0.5 * self.env.Atmospheric_Density * self.env.Drag * \
                      self.env.Reference_Area * np.linalg.norm(v_body) * v_body

        # Calculate torque in Body frame (Shape_Vector is already in Body frame)
        T_drag_body = np.cross(self.Shape_Vector, F_drag_body)
        return T_drag_body
    
    def calculate_gravity_gradient_torque_body(self, current_q_eci_to_body: Quaternion, current_r_eci: np.ndarray):
        """
        Calculates gravity gradient torque and expresses it in the Body Frame.
        Args:
            current_q_eci_to_body (Quaternion): Current attitude of Body relative to ECI.
            current_r_eci (np.ndarray): Current position of spacecraft in ECI.
        Returns:
            np.ndarray: Gravity gradient torque vector in the Body Frame (N*m).
        """
        # Transform ECI position vector to Body frame
        r_body = current_q_eci_to_body.rotate(current_r_eci)
        
        # Unit vector from spacecraft to Earth's center in Body frame (nadir vector)
        n_body = -r_body / np.linalg.norm(r_body)

        # Gravity gradient torque formula (in Body Frame)
        mu = 3.986004418e14 # Earth's standard gravitational parameter in m^3/s^2 (Poliastro's Earth.k)
        
        # T_gg_body = 3 * (mu / ||r||^3) * (n_body x (I_body @ n_body))
        T_gg_body = 3 * mu / (np.linalg.norm(r_body)**3) * np.cross(n_body, self.Inertial_Matrix @ n_body)
        
        return T_gg_body
    
    def calculate_torque_noise_body(self):
        """
        Generates noise torque components directly in the Body Frame.
        Returns:
            np.ndarray: Noise torque vector in the Body Frame (N*m).
        """
        noise_torque = np.random.normal(self.env.noise_mean, self.env.noise_std, 3)
        return noise_torque

    def attitude_dynamics(self, time_s, y, control_torque_body: np.ndarray):
        """
        Differential equations for spacecraft attitude dynamics (quaternion and angular velocity).
        All calculations are consistently in the Body Frame for torques and angular velocity.
        Args:
            time_s (float): Current simulation time in seconds.
            y (np.ndarray): Current state vector [q0, q1, q2, q3, wx, wy, wz].
            control_torque_body (np.ndarray): Control torque vector from the agent, in Body frame (N*m).
        Returns:
            np.ndarray: Derivatives [q_dot0, q_dot1, q_dot2, q_dot3, w_dotx, w_doty, w_dotz].
        """
        # Extract current state from y
        current_q = Quaternion(y[0:4]) # Current attitude Q_ECI_to_Body
        current_w_body = y[4:7] # Current angular velocity in Body frame

        # Get current orbital position and velocity in ECI
        r_eci, v_eci = self.get_orbit_state_at_time(time_s)

        # Calculate individual torques, ensuring they are in the Body Frame
        total_disturbance_torque_body = np.zeros(3)
        
        total_disturbance_torque_body += self.calculate_gravity_gradient_torque_body(current_q, r_eci)
        total_disturbance_torque_body += self.calculate_aerodynamic_torque_body(current_q, v_eci)
        total_disturbance_torque_body += self.calculate_torque_noise_body()

        # Sum all torques: disturbances + control torque
        total_torque_on_body = total_disturbance_torque_body + control_torque_body

        # Euler's Equation for angular velocity derivative (in Body Frame)
        w_dot_body = np.linalg.inv(self.Inertial_Matrix) @ \
                     (total_torque_on_body - np.cross(current_w_body, self.Inertial_Matrix @ current_w_body))

        # Quaternion Kinematic Equation
        # q_dot = 0.5 * q * omega_quat (where omega_quat is [0, wx, wy, wz])
        q_dot_quat = 0.5 * current_q * Quaternion(0, current_w_body[0], current_w_body[1], current_w_body[2])
        
        return np.concatenate([q_dot_quat.elements, w_dot_body])
    
    def solver(self, time_span_tuple, y_initial, control_torque_body: np.ndarray):
        """
        Solves the attitude dynamics over a specified time span with a given control torque.
        Args:
            time_span_tuple (tuple): A tuple (t_start, t_end) for the integration.
            y_initial (np.ndarray): Initial state vector for the solver.
            control_torque_body (np.ndarray): Constant control torque to apply during this span.
        Returns:
            scipy.integrate.OdeResult: The solution object.
        """
        # Note: The `args` parameter passes arguments to the `attitude_dynamics` function
        sol = solve_ivp(self.attitude_dynamics, time_span_tuple, y_initial,
                        args=(control_torque_body,), # Pass control torque here
                        dense_output=True,
                        max_step=(time_span_tuple[1] - time_span_tuple[0]) / 10.0, # Smaller internal steps
                        method='RK45')
        return sol