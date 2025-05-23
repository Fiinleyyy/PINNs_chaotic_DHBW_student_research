import numpy as np
import tensorflow as tf

from scipy.integrate import solve_ivp

def ref_solution(A, B, C, t_min, t_max, initial_conditions):
    def lorenz_rhs(t, y):
        x, y_, z = y
        return [A*(y_ - x),
                x*(B - z) - y_,
                x*y_ - C*z]
    t_span = (t_min, t_max)
    t_eval = np.linspace(t_min, t_max, 1000)
    sol = solve_ivp(
        lorenz_rhs,
        t_span,
        initial_conditions,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
        dense_output=True
    )
    return t_eval, sol

def generate_noisy_data(sol, t_min, t_max):
    """
    Generates noisy data points from the reference solution.
    Returns t_data and xyz_data as tensors.
    """
    n_data = 100
    t_data_np = np.linspace(t_min, t_max, n_data).reshape(-1, 1)
    xyz_np = sol.sol(t_data_np.flatten()).T
    noise = 0.1 * np.random.randn(*xyz_np.shape)
    xyz_noisy = xyz_np + noise

    t_data = tf.convert_to_tensor(t_data_np, dtype=tf.float32)
    xyz_data = tf.convert_to_tensor(xyz_noisy, dtype=tf.float32)
    return t_data, xyz_data

def generate_noisy_data_with_gap(sol, t_min, t_max, gap_start, gap_end):
    t_data, xyz_data = generate_noisy_data(sol, t_min, t_max)

    t_np = t_data.numpy().flatten()
    xyz_np = xyz_data.numpy()

    t_kept = []
    xyz_kept = []
    for t, xyz in zip(t_np, xyz_np):
        if t < gap_start or t > gap_end:
            t_kept.append([t])
            xyz_kept.append(xyz)       

    t_data_gap = tf.convert_to_tensor(t_kept, dtype=tf.float32)
    xyz_data_gap = tf.convert_to_tensor(xyz_kept, dtype=tf.float32)

    return t_data_gap, xyz_data_gap

def generate_partial_noisy_data(y_data, column='x'):
    match column:
        case 'x': 
            y_partial_data = y_data[:, 0:1]
        case 'y': 
            y_partial_data = y_data[:, 1:2]
        case 'z': 
            y_partial_data = y_data[:, 2:3]
    print("Test")
    return y_partial_data

