import numpy as np 

class Orbit_Parameters:
    SEMI_MAJOR_AXIS = 6871.0 # This is the key
    ECCENTRICITY  = 0.001
    INCLINATION = 45.0
    RAAN = 0.0
    ARGP = 0.0
    TRUE_ANOMALY = 0.0
    DRAG = 2.2
    REFERENCE_AREA = 1.0
    ATMOSPHERIC_DENSITY = 1e-12
    NOISE_MEAN = 0.0
    NOISE_STD = 1e-9
    INITIAL_ATTITUDE = np.array([1,0,0])
    INITIAL_ANGLE = np.deg2rad(10.0)
    SHAPE_VECTOR = np.array([0.5, 0.1, 0.2])
    INERTIAL_MATRIX = np.array([
                                    [100.0, 0.0, 0.0],
                                    [0.0, 100.0, 0.0],
                                    [0.0, 0.0, 100.0]
                                ])
    INITIAL_LINEAR_VELOCITY = np.array([0.0, 7615.0, 0.0])
    INITIAL_ANGULAR_VELOCITY = np.array([10, 0.0, 0.0])


    
