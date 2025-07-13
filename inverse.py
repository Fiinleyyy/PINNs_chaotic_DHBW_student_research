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
    # Konvertiere Zeit und berechne PINN-Vorhersage
    t_plot = tf.convert_to_tensor(t_eval.reshape(-1, 1), dtype=tf.float32)
    y_pred = model(t_plot).numpy()

    # Plot-Setup
    plt.figure(figsize=(12, 8))
    labels = ['x', 'y', 'z']

    # Subplots für x, y, z
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_eval, sol.y[i], 'k-', label='RK45 (True)')
        plt.plot(t_eval, y_pred[:, i], 'r--', label='PINN Prediction')
        plt.ylabel(labels[i])

        # Titel mit optionalen Parametern
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

    # Parameter-Ausgabe (optional)
    print("##################################")
    if true_params is not None:
        print(f"Ground truth Parameter: A={true_params[0]}, B={true_params[1]}, C={true_params[2]}")
    if predicted_params is not None:
        print(f"Geschätzte Parameter: A={predicted_params[0].numpy():.4f}, "
              f"B={predicted_params[1].numpy():.4f}, C={predicted_params[2].numpy():.4f}")

    plt.show()

# ──────────────── Main Function ────────────────
def main():
    # PINN Architektur
    HIDDEN_LAYER = 6
    NEURONS_PER_LAYER = 30
    ACTIVATION_FUNCTION = tf.keras.activations.silu
    WEIGHT_INITIALIZATION = tf.keras.initializers.GlorotUniform

    # Trainings-Hyperparameter
    LEARNING_RATE = 0.01

    DECAY_RATE = 0.8
    OPTIMIZER = tf.keras.optimizers.Adam
    EPOCHS = 25000
    COLLOCATION_POINTS = 4024
    ALPHA_DATA = 0.5
    NORMALIZE_INPUT = True
    DATA_ACTIVE = True

    # Time domain
    t_min, t_max = 0.0, 15.0

    # Trainable parameters
    A = tf.Variable(10.0, dtype=tf.float32, trainable=False, name="A")
    B = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="B")
    C = tf.Variable(2.667, dtype=tf.float32, trainable=False, name="C")

    # System parameters (cusotmizable by programmer)
    True_A, True_B, True_C = 10, 0.5, 8/3
    INITIAL_CONDITIONS = np.array([1, 1, 1], dtype=np.float32)

    # Create reference and noisy data
    t_eval, sol = hf.ref_solution(True_A, True_B, True_C, t_min, t_max, INITIAL_CONDITIONS)
    t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max, 0.3)

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
        trainable_parameters = [B],
        chaotic=True
    )

    # Evaluate and plot
    evaluate_and_plot(model, t_eval, sol, (A, B, C), (True_A, True_B, True_C))


if __name__ == "__main__":
    main()
