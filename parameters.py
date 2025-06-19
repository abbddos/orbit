import numpy as np 

class Orbit_Parameters:
    SEMI_MAJOR_AXIS = 6871.0
    ECCENTRICITY  = 0.001
    INCLINATION = 45.0
    RAAN = 0.0
    ARGP = 0.0
    TRUE_ANOMALY = 0.0  #nu 
    DRAG = 2.2
    REFERENCE_AREA = 1.0
    ATMOSPHERIC_DENSITY = 1e-12
    NOISE_MEAN = 0.0
    NOISE_STD = 1e-9
    INITIAL_ATTITUDE = np.array([1,0,0])
    INITIAL_ANGLE = np.deg2rad(10.0)
    SHAPE_VECTOR = np.array([0.05, 0.01, 0.02])
    INERTIAL_MATRIX = np.array([
                                    [10.0, 0.0, 0.0],
                                    [0.0, 20.0, 0.0],
                                    [0.0, 0.0, 30.0]
                                ]) 
    INITIAL_LINEAR_VELOCITY = np.array([0.0, 7615.0, 0.0]) #V
    INITIAL_ANGULAR_VELOCITY = np.array([0.001, 0.0, 0.0]) #W


    
