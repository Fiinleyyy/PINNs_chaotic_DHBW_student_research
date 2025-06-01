import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

import pinn_helper_functions as phf
import helper_functions as hf

def set_seed(seed=42):
    """
    Sets the global seed for reproducibility across random, numpy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_train_step():
    """
    Creates a TensorFlow function for performing a single training step.
    Computes losses, gradients, and updates model weights and Lorenz parameters.
    Returns the total loss, physics loss, and data/initial condition loss.
    """
    # @tf.function decorator is wrapped by a method, otherwise creating more than one model instance won't be allowed by TensorFlow
    @tf.function
    def train_step(model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input):
        """
        Performs a single training step: computes losses, gradients, and updates the model weights and Lorenz parameters.
        Returns the total loss, physics loss, and data / initial condition loss.
        """
        with tf.GradientTape() as tape:
            # Compute physics loss and scale it
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            loss_phys_scaled = loss_phys / tf.reduce_mean(loss_phys + 1e-8)  # Normalization

            if data_active:
                # Compute data loss
                loss_data = phf.data_loss(model, t_data, y_data)
                # Combine the losses with dynamic alpha
                loss = alpha * loss_data + (1 - alpha) * loss_phys_scaled
            else:
                # Compute initial condition loss
                loss_ic = phf.initial_condition_loss(model, t_initial, initial_conditions, t_min, t_max, normalize_input)
                loss = alpha * loss_ic + (1 - alpha) * loss_phys_scaled

            # Compute gradients and apply gradient clipping
            grads = tape.gradient(loss, model.trainable_variables)
            clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]  # Clipping
            model.optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

            # Return the losses
            if data_active:
                return loss, loss_data, loss_phys
            else:
                return loss, loss_ic, loss_phys
    return train_step

# ──────────────── Train function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, data_active, t_data, y_data):
    """
    Trains the PINN model using physics-informed and data-driven losses.
    Dynamically adjusts alpha and resamples collocation points during training.
    Outputs training progress every 1000 epochs.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)
    model.optimizer = optimizer_class(learning_rate=lr_schedule)

    # Use the train_step function from create_train_step
    train_step = create_train_step()

    print("Training started...")

    for epoch in range(epochs):
        # Dynamic adjustment of alpha
        dynamic_alpha = max(0.1, alpha * (1 - epoch / epochs))  # Reduce alpha over time

        # Resample collocation points
        t_collocation = phf.sample_collocation(t_min, t_max, collocation_points, normalize_input)

        # Perform a training step
        step_loss, ic_or_data_loss, phy_loss = train_step(
            model, t_initial, initial_conditions, t_collocation, dynamic_alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input
        )

        # Output every 1000 epochs
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Loss: {step_loss:.4e} | Data/IC-Loss: {ic_or_data_loss:.4e} | Physics-Loss: {phy_loss:.4e}")

    print("Training finished!")

def pinn_predict(model, t_eval, t_min, t_max, normalize_input):
    """
    Predicts the solution of the Lorenz system using the trained PINN model.
    Normalizes input time values if specified.
    Returns the predicted values as a numpy array.
    """
    if normalize_input:
        t_norm = phf.normalize_time(t_eval.reshape(-1,1), t_min, t_max)
    else:
        t_norm = t_eval.reshape(-1,1)
    t_plot = tf.convert_to_tensor(t_norm, dtype=tf.float32)
    return model(t_plot).numpy()

def plot_results(t_eval, sol, y_pinn):
    """
    Plots the comparison between the reference solution (RK45) and the PINN predictions.
    Displays results for x, y, and z components of the Lorenz system.
    """
    # NOTE this function is not needed for Jupyter notebook, just in case if pinn.py is desired to be run directly
    plt.figure(figsize=(12,8))
    labels = ['x','y','z']
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t_eval, sol.y[i],  'k-', label="RK45 (Reference)")
        plt.plot(t_eval, y_pinn[:,i],'r--', label="PINN")
        plt.ylabel(labels[i])
        if i==0:
            plt.title("Comparison PINN vs. RK45 for the Lorenz system")
        if i==2:
            plt.xlabel("t")
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# ──────────────── Main Routine ──────────────
# NOTE: not needed for Jupyter notebook, remove later on
def main():
    """
    Main routine for training and evaluating the PINN model.
    Sets the seed, defines system parameters, builds the model, trains it, and plots results.
    """
    set_seed(42)
    # System parameters
    A, B, C = 10, 28, 8/3  # Lorenz system parameters
    INITIAL_CONDITIONS = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Time domain
    t_min, t_max = 0.0, 10.0

    # PINN architecture
    HIDDEN_LAYER = 6
    NEURONS_PER_LAYER = 30  # Increase the number of neurons
    ACTIVATION_FUNCTION = tf.keras.activations.silu
    WEIGHT_INITIALIZATION = tf.keras.initializers.HeNormal  # Use He-Normal initialization

    # Training hyperparameters
    LEARNING_RATE = 0.01  # Reduce the initial learning rate
    DECAY_RATE = 0.8  # Slow down the decay rate of the learning rate
    OPTIMIZER = tf.keras.optimizers.Adam
    EPOCHS = 5000  # Increase the number of epochs
    COLLOCATION_POINTS = 4000  # Increase the number of collocation points
    ALPHA_DATA = 0.1  # Give more weight to physics loss
    NORMALIZE_INPUT = True
    DATA_ACTIVE = True

    # Create reference solution
    t_eval, sol = hf.ref_solution(A, B, C, t_min, t_max, INITIAL_CONDITIONS)

    # Add noise to reference solution
    try:
        t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max, noise_std=0.01)
    except TypeError:
        t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max)

    if NORMALIZE_INPUT:
        t_data = phf.normalize_time(t_data, t_min, t_max)

    # Build model
    model = phf.build_pinn_network(HIDDEN_LAYER, NEURONS_PER_LAYER, ACTIVATION_FUNCTION, WEIGHT_INITIALIZATION)

    # Train
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
        data_active=DATA_ACTIVE,
        t_data=t_data,
        y_data=y_data
    )

    # PINN prediction
    y_pinn = pinn_predict(model, t_eval, t_min, t_max, normalize_input=NORMALIZE_INPUT)

    # Plot
    plot_results(t_eval, sol, y_pinn)

if __name__ == "__main__":
    """
    Executes the main routine if the script is run directly.
    """
    main()
