import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import helper_functions as hf
import pinn_helper_functions as phf

import logging

logging.basicConfig(
    filename="training_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ──────────────── Training Step ────────────────
def create_train_step():
    @tf.function
    def train_step(model, t0, y0, t_collocation, t_data, y_data, alpha, A, B, C, t_min, t_max, normalize_input, trainable_parameters):
        with tf.GradientTape() as tape:
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            loss_data = phf.data_loss(model, t_data, y_data)
            total_loss = (1 - alpha) * loss_phys + alpha * loss_data

        variables = model.trainable_variables + trainable_parameters
        grads = tape.gradient(total_loss, variables)
        model.optimizer.apply_gradients(zip(grads, variables))
        return total_loss, loss_data, loss_phys,
    return train_step

# ──────────────── Training Function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, t_data, y_data, trainable_parameters):
    print(B)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)
    model.optimizer = optimizer_class(learning_rate=lr_schedule)

    train_step = create_train_step()

    print("Training started...")
    for epoch in range(epochs):
        t_collocation = phf.sample_collocation(t_min, t_max, collocation_points, normalize_input)
        step_loss, data_loss, phy_loss = train_step(
            model,
            t_initial,
            initial_conditions,
            t_collocation,
            t_data,
            y_data,
            alpha,
            A,
            B,
            C,
            t_min,
            t_max,
            normalize_input,
            trainable_parameters
        )
        if epoch % 10 == 0:
            log_entry = (
             f"Epoch {epoch}: Total={step_loss}, Physics={phy_loss}, Data={data_loss}, A={A.numpy():.4f}, B={B.numpy():.4f}, C={C.numpy():.4f}"
            )
            logging.info(log_entry)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total={step_loss}, Physics={phy_loss}, Data={data_loss}, A={A.numpy():.4f}, B={B.numpy():.4f}, C={C.numpy():.4f}")

# ──────────────── Evaluation Function ────────────────
def evaluate_and_plot(model, t_eval, sol, predicted_params, true_params):
    A, B, C = predicted_params
    true_A, true_B, true_C = true_params
    t_plot = tf.convert_to_tensor(t_eval.reshape(-1, 1), dtype=tf.float32)
    y_pred = model(t_plot).numpy()

    plt.figure(figsize=(12, 8))
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
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

# ──────────────── Main Function ────────────────
def main():
    # PINN Architektur
    HIDDEN_LAYER = 6
    NEURONS_PER_LAYER = 30
    ACTIVATION_FUNCTION = tf.keras.activations.silu
    WEIGHT_INITIALIZATION = tf.keras.initializers.GlorotUniform

    # Trainings-Hyperparameter
    LEARNING_RATE = 0.05

    DECAY_RATE = 0.09
    OPTIMIZER = tf.keras.optimizers.Adam
    EPOCHS = 5000
    COLLOCATION_POINTS = 1024
    ALPHA_DATA = 0.5
    NORMALIZE_INPUT = False
    DATA_ACTIVE = True

    # Time domain
    t_min, t_max = 0.0, 10.0

    # Trainable parameters
    A = tf.Variable(10.0, dtype=tf.float32, trainable=False, name="A")
    B = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="B")
    C = tf.Variable(2.667, dtype=tf.float32, trainable=False, name="C")

    # System parameters (cusotmizable by programmer)
    True_A, True_B, True_C = 10, 5, 8/3
    INITIAL_CONDITIONS = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Create reference and noisy data
    t_eval, sol = hf.ref_solution(True_A, True_B, True_C, t_min, t_max, INITIAL_CONDITIONS)
    t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max, 0.1)

    # Build model
    model = phf.build_pinn_network(HIDDEN_LAYER, NEURONS_PER_LAYER, ACTIVATION_FUNCTION, WEIGHT_INITIALIZATION)

    # Train model
    train(
        model,
        t_initial=t_min,
        initial_conditions=INITIAL_CONDITIONS,
        A=A, B=B, C=C,
        t_min=t_min, t_max=t_max,
        collocation_points=COLLOCATION_POINTS,
        alpha=ALPHA_DATA,
        learning_rate=LEARNING_RATE,
        decay_rate=DECAY_RATE,
        epochs=EPOCHS,
        optimizer_class=OPTIMIZER,
        normalize_input=NORMALIZE_INPUT,
        t_data=t_data,
        y_data=y_data,
        trainable_parameters = [B]
    )

    # Evaluate and plot
    evaluate_and_plot(model, t_eval, sol, (A, B, C), (True_A, True_B, True_C))


if __name__ == "__main__":
    main()
