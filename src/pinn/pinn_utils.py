import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

def lorenz_system(x, y, z, A, B, C):
    """
    Calculates the derivatives of the Lorenz system for given x, y, z and current parameters A, B, C.
    Returns dx/dt, dy/dt, dz/dt.
    """
    dxdt = A * (y - x)
    dydt = x * (B - z) - y
    dzdt = x * y - C * z
    return dxdt, dydt, dzdt

def build_pinn_network(hidden_layer, neurons_per_layer, activation_function, weight_initializer):
    """
    Builds and returns the neural network model for the PINN.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,), name="input"))
    
    for i in range(hidden_layer):
        model.add(tf.keras.layers.Dense(
            units=neurons_per_layer,
            activation=activation_function,
            kernel_initializer=weight_initializer(),
            name=f"hidden_{i}"))

    model.add(tf.keras.layers.Dense(units=3, activation=None, kernel_initializer=weight_initializer(), name="output"))
    #model.summary()
    return model

def normalize_time(t, t_min, t_max):
    "Normalizes and returns time input vector"
    return (t - t_min) / (t_max - t_min)

def sample_collocation(t_min, t_max, collocation_points, normalize_input):
    """
    Samples random collocation points in the time domain for physics loss calculation.
    Returns a tensor of shape (COLLOCATION_POINTS, 1).
    """
    t_raw = t_min + (t_max - t_min) * np.random.rand(collocation_points)
    t_collocation = normalize_time(t_raw, t_min, t_max) if normalize_input else t_raw
    t_collocation = np.expand_dims(t_collocation, axis=1)
    return tf.convert_to_tensor(t_collocation, dtype=tf.float32)

# ──────────────── Loss functions ────────────────
def physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input):
    """
    Calculates the physics loss by comparing the model's derivatives with the Lorenz system equations at the collocation points.
    Returns the mean squared error of the residuals.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_collocation)
        pred = model(t_collocation)
        x, y, z = tf.split(pred, num_or_size_splits=3, axis=1)

    dx_dt_hat = tape.gradient(x, t_collocation)
    dy_dt_hat = tape.gradient(y, t_collocation)
    dz_dt_hat = tape.gradient(z, t_collocation)

    del tape

    if normalize_input:
        scale = t_max - t_min
        dx_dt = dx_dt_hat / scale
        dy_dt = dy_dt_hat / scale
        dz_dt = dz_dt_hat / scale
    else:
        dx_dt = dx_dt_hat
        dy_dt = dy_dt_hat
        dz_dt = dz_dt_hat

    dx_dt_true, dy_dt_true, dz_dt_true = lorenz_system(x, y, z, A, B, C)

    return tf.reduce_mean((dx_dt - dx_dt_true)**2 +
                          (dy_dt - dy_dt_true)**2 +
                          (dz_dt - dz_dt_true)**2)

def data_loss(model, t_data, y_data):
    """
    Calculates the mean squared error between the model's prediction and the noisy data.
    Returns the data loss.
    """
    pred = model(t_data)
    return tf.reduce_mean((pred - y_data)**2)

def initial_condition_loss(model, t_initial, initial_conditions, t_min, t_max, normalize_input):
    """
    Calculates and returns mean squared error between predicted inital condition values and actually
    initial condition values.
    """
    if normalize_input:
        t_initial_norm = normalize_time(t_initial, t_min, t_max)
    else:
        t_initial_norm = t_initial
    t_initial_tensor = tf.constant([[t_initial_norm]], dtype=tf.float32)
    initial_conditions_pred = model(t_initial_tensor)
    true_initial_conditions = tf.reshape(initial_conditions, (1, -1))
    return tf.reduce_mean((true_initial_conditions - initial_conditions_pred)**2)