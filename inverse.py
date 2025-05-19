import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ──────────────── Global configuration ────────────────
# Lorenz system parameter // tf.Variable so the variables are adjustable during training
# Initial values are set to 1 - might need to adjust
A = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="A")
B = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="B")
C = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="C")

# Initial conditions (later: chaotic / not chaotic)
INITIAL_CONDITIONS = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# PINN architecture
HIDDEN_LAYER = 6
NEURONS_PER_LAYER = 30
ACTIVATION_FUNCTION = tf.keras.activations.silu # NOTE: Silu equals swish activation function
WEIGHT_INITIALIZATION = tf.keras.initializers.GlorotUniform

# Hyperparameters for training
LEARNING_RATE = 0.01
DECAY_RATE = 0.09
OPTIMIZER = tf.keras.optimizers.Adam
EPOCHS = 5000 #25000
COLLOCATION_POINTS = 1024
ALPHA_DATA = 0.5

# Domain
t_min, t_max = 0.0, 10.0


# ──────────────── Lorenz system ────────────────
def lorenz_system(x, y, z):
    """
    Calculates the derivatives of the Lorenz system for given x, y, z and current parameters A, B, C.
    Returns dx/dt, dy/dt, dz/dt.
    """
    dxdt = A * (y - x)
    dydt = x * (B - z) - y
    dzdt = x * y - C * z
    return dxdt, dydt, dzdt


# ──────────────── Build network ────────────────
def build_network():
    """
    Builds and returns the neural network model for the PINN.
    """
    # NOTE: seeds?
    model = tf.keras.Sequential()

    # build input layers
    model.add(tf.keras.layers.InputLayer(input_shape=(1,), name="input"))

    # build hidden layers
    for i in range(HIDDEN_LAYER):
        model.add(tf.keras.layers.Dense(units=NEURONS_PER_LAYER, activation=ACTIVATION_FUNCTION, kernel_initializer=WEIGHT_INITIALIZATION(), name=f"hidden_{i}"))

    # build output layer (x,y,z)
    model.add(tf.keras.layers.Dense(units=3, activation=None, kernel_initializer=WEIGHT_INITIALIZATION(), name="output"))
    model.summary()
    return model

# ──────────────── Loss functions ────────────────
def sample_collocation():
    """
    Samples random collocation points in the time domain for physics loss calculation.
    Returns a tensor of shape (COLLOCATION_POINTS, 1).
    """
    t_collocation = t_min + (t_max - t_min) * np.random.rand(COLLOCATION_POINTS)
    t_collocation = np.expand_dims(t_collocation, axis=1)
    t_collocation = tf.convert_to_tensor(t_collocation, dtype=tf.float32)
    return t_collocation

def physics_loss(model, t_collocation):
    """
    Calculates the physics loss by comparing the model's derivatives with the Lorenz system equations at the collocation points.
    Returns the mean squared error of the residuals.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_collocation)
        pred = model(t_collocation)  

        x, y, z = tf.split(pred, num_or_size_splits=3, axis=1)

    dx_dt = tape.gradient(x, t_collocation)
    dy_dt = tape.gradient(y, t_collocation)
    dz_dt = tape.gradient(z, t_collocation)
    del tape

    # Lorenz-System Gleichungen
    dx_dt_true, dy_dt_true, dz_dt_true = lorenz_system(x, y, z)

    physics_loss = tf.reduce_mean((dx_dt - dx_dt_true)**2 +
                        (dy_dt - dy_dt_true)**2 +
                        (dz_dt - dz_dt_true)**2)
    return physics_loss

# def initial_condition_loss(model, t_initial, initial_conditions):
#     t_initial_tensor = tf.constant([[t_initial]], dtype = tf.float32)
#     initial_conditions_pred = model(t_initial_tensor)

#     true_initial_conditions = tf.reshape(initial_conditions, (1, -1))

#     initial_condition_loss = tf.reduce_mean((true_initial_conditions - initial_conditions_pred)**2)
#     return initial_condition_loss

def data_loss(model, t_data, y_data):
    """
    Calculates the mean squared error between the model's prediction and the noisy data.
    Returns the data loss.
    """
    pred = model(t_data)
    return tf.reduce_mean((pred - y_data)**2)

# ──────────────── Training step ────────────────
@tf.function
def train_step(model, t0, y0, t_collocation, t_data, y_data, alpha):
    """
    Performs a single training step: computes losses, gradients, and updates the model weights and Lorenz parameters.
    Returns the total loss, physics loss, and data loss.
    """
    with tf.GradientTape() as tape:
        #loss_ic = initial_condition_loss(model, t0, y0) // not needed in reverse problem
        loss_phys = physics_loss(model, t_collocation)
        loss_data = data_loss(model, t_data, y_data)

        total_loss = (1 - alpha) * loss_phys + alpha * loss_data

    # adding searched for variables to the list of adjustable weights and biases during backpropagation
    variables = model.trainable_variables + [A, B, C]
    grads = tape.gradient(total_loss, variables)
    model.optimizer.apply_gradients(zip(grads, variables))
    return total_loss, loss_phys, loss_data

# ──────────────── Training function ────────────────
def train(model, t0, y0, t_data, y_data):
    """
    Trains the PINN model and Lorenz parameters over multiple epochs using the specified optimizer and learning rate schedule.
    Prints progress and loss values during training.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=1000,
        decay_rate=DECAY_RATE)

    model.optimizer = OPTIMIZER(learning_rate=lr_schedule)
    for epoch in range(EPOCHS):
        t_collocation = sample_collocation()
        loss, phys, data = train_step(model, t0, y0, t_collocation, t_data, y_data, ALPHA_DATA)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total={loss}, Physics_loss={phys}, Data_loss={data:}, A={A.numpy()}, B={B.numpy()}, C={C.numpy()}")

# ──────────────── Data generation and evaluation ────────────────
def generate_reference_data():
    """
    Generates RK45 reference data (ground truth) for the Lorenz system.
    Returns t_eval and the solution object.
    """
    true_A, true_B, true_C = 10, 20, 2.6667

    def rhs(t, y):
        x, y_, z = y
        return [true_A * (y_ - x),
                x * (true_B - z) - y_,
                x * y_ - true_C * z]

    t_eval = np.linspace(t_min, t_max, 1000)
    sol = solve_ivp(rhs, (t_min, t_max), INITIAL_CONDITIONS, t_eval=t_eval, rtol=1e-9, atol=1e-9, dense_output=True)
    return t_eval, sol, true_A, true_B, true_C

def generate_noisy_data(sol):
    """
    Generates noisy data points from the reference solution.
    Returns t_data and xyz_data as tensors.
    """
    n_data = 100
    t_data_np = np.linspace(t_min, t_max, n_data).reshape(-1, 1)
    xyz_np = sol.sol(t_data_np.flatten()).T
    noise = 0.01 * np.random.randn(*xyz_np.shape)
    xyz_noisy = xyz_np + noise

    t_data = tf.convert_to_tensor(t_data_np, dtype=tf.float32)
    xyz_data = tf.convert_to_tensor(xyz_noisy, dtype=tf.float32)
    return t_data, xyz_data

def evaluate_and_plot(model, t_eval, sol, true_A, true_B, true_C):
    """
    Evaluates the trained model and plots the results compared to the reference solution.
    Also prints the estimated and true parameters.
    """
    t_plot = tf.convert_to_tensor(t_eval.reshape(-1,1), dtype=tf.float32)
    y_pred = model(t_plot).numpy()

    plt.figure(figsize=(12, 8))
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_eval, sol.y[i], 'k-', label='RK45 (True)')
        plt.plot(t_eval, y_pred[:, i], 'r--', label='PINN Prediction')
        plt.ylabel(labels[i])
        if i == 0:
            plt.title("Inverse PINN für das Lorenz-System")
        if i == 2:
            plt.xlabel("t")
        plt.legend()
    plt.tight_layout()
    print("##################################")
    print(f"Ground truth Parameter: A={true_A}, B={true_B}, C={true_C}")
    print(f"Geschätzte Parameter: A={A.numpy():.4f}, B={B.numpy():.4f}, C={C.numpy():.4f}")
    plt.show()

def main():
    """
    Main routine: generates data, builds and trains the model, evaluates predictions, and plots results.
    """
    # 1) Generate reference data
    t_eval, sol, true_A, true_B, true_C = generate_reference_data()

    # 2) Generate noisy data
    t_data, xyz_data = generate_noisy_data(sol)

    # 3) Build and train model
    model = build_network()
    train(model, t0=t_min, y0=INITIAL_CONDITIONS, t_data=t_data, y_data=xyz_data)

    # 4) Evaluate and plot results
    evaluate_and_plot(model, t_eval, sol, true_A, true_B, true_C)

if __name__ == "__main__":
    main()


