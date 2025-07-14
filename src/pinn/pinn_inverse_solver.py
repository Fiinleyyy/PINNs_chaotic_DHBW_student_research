import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from . import pinn_utils as phf

import logging

logging.basicConfig(
    filename="training_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ──────────────── Training Step ────────────────
def create_train_step():
    """Defines the train step of each epoch in training of PINN model regarding the inverse problem."""
    @tf.function
    def train_step(model, t0, y0, t_collocation, t_data, y_data, alpha, A, B, C, t_min, t_max, normalize_input, trainable_parameters, chaotic=False):
        with tf.GradientTape() as tape:
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            if chaotic:
                loss_phys = loss_phys / tf.reduce_mean(loss_phys + 1e-8)  # Normalisierung
            loss_data = phf.data_loss(model, t_data, y_data)
            total_loss = (1 - alpha) * loss_phys + alpha * loss_data
        variables = model.trainable_variables + trainable_parameters
        grads = tape.gradient(total_loss, variables)
        model.optimizer.apply_gradients(zip(grads, variables))
        return total_loss, loss_data, loss_phys,
    return train_step

# ──────────────── Training Function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, t_data, y_data, trainable_parameters, chaotic=False):
    """Perfoms training of a PINN model regarding the inverse problem."""
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
            trainable_parameters,
            chaotic
        )
        if epoch % 10 == 0:
            log_entry = (
             f"Epoch {epoch}: Total={step_loss}, Physics={phy_loss}, Data={data_loss}, A={A.numpy():.4f}, B={B.numpy():.4f}, C={C.numpy():.4f}"
            )
            logging.info(log_entry)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total={step_loss}, Physics={phy_loss}, Data={data_loss}, A={A.numpy():.4f}, B={B.numpy():.4f}, C={C.numpy():.4f}")

# ──────────────── Evaluation Function ────────────────
def evaluate_and_plot(model, t_eval, sol, predicted_params=None, true_params=None, initial_conditions=None):
    """Plots PINN prediction for the inverse problem and predicted parameters."""
    # Convert time and calculate PINN prediction
    t_plot = tf.convert_to_tensor(t_eval.reshape(-1, 1), dtype=tf.float32)
    y_pred = model(t_plot).numpy()

    # Plot-Setup
    plt.figure(figsize=(12, 8))
    labels = ['x', 'y', 'z']

    # Subplots for x, y, z
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_eval, sol.y[i], 'k-', label='RK45 (True)')
        plt.plot(t_eval, y_pred[:, i], 'r--', label='PINN Prediction')
        plt.ylabel(labels[i])

        # Title
        if i == 0:
            if true_params is not None and initial_conditions is not None:
                ic_str = ', '.join([f'{v:.3f}' for v in initial_conditions])
                title = f"Inverse PINN – IC: [{ic_str}], True $\\rho$ = {true_params[1]}"
            else:
                title = "Inverse PINN für das Lorenz-System"
            plt.title(title)

        if i == 2:
            plt.xlabel("t")

        plt.legend()

    plt.tight_layout()

    # Parameter display in the plot
    print("##################################")
    if true_params is not None:
        print(f"Ground truth Parameter: A={true_params[0]}, B={true_params[1]}, C={true_params[2]}")
    if predicted_params is not None:
        print(f"Geschätzte Parameter: A={predicted_params[0].numpy():.4f}, "
              f"B={predicted_params[1].numpy():.4f}, C={predicted_params[2].numpy():.4f}")

    plt.show()