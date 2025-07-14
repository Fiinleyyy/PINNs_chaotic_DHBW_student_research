import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import pinn_utils as phf

def create_train_step():
    """Creates and defines the train step of each epoch in training of PINN model regarding the forward problem."""
    # @tf.function decorater is wrapped by a method, otherwise creating more than one model instance won't be allowed by tf
    @tf.function
    def train_step(model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input, chaotic):
        """
        Performs a single training step: computes losses, gradients, and updates the model weights and Lorenz parameters.
        Returns the total loss, physics loss, and data / initial condition loss.
        """
        with tf.GradientTape() as tape:
            # Caluclate physics loss
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            
            if chaotic:
                loss_phys = loss_phys / tf.reduce_mean(loss_phys + 1e-8)  # Normalisierung

            if data_active:
                # Calculate data loss
                loss_data = phf.data_loss(model, t_data, y_data)
                # Combine main loss function
                loss = alpha * loss_data + (1 - alpha) * loss_phys
            else:
                # Calculate IC loss
                loss_ic = phf.initial_condition_loss(model, t_initial, initial_conditions, t_min, t_max, normalize_input)
                loss = alpha * loss_ic + (1 - alpha) * loss_phys

            # Calculate gradients
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Return the calculated losses
            if data_active:
                return loss, loss_data, loss_phys
            else:
                return loss, loss_ic, loss_phys
    return train_step

# ──────────────── Train function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, data_active, t_data, y_data, chaotic=False):
    """Perfoms training of a PINN model regarding the forward problem."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)
    model.optimizer = optimizer_class(learning_rate=lr_schedule)

    # Create a training step
    train_step = create_train_step()

    print("Training started...")

    for epoch in range(epochs):
        # Resample collocation points
        t_collocation = phf.sample_collocation(t_min, t_max, collocation_points, normalize_input)

        # Perform a train step
        step_loss, ic_or_data_loss, phy_loss = train_step(
            model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input, chaotic
        )

        # Log loss results every 1000 epochs
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Loss: {step_loss:.4e} | Data/IC-Loss: {ic_or_data_loss:.4e} | Physics-Loss: {phy_loss:.4e}")

    print("Training finished!")

def pinn_predict(model, t_eval, t_min, t_max, normalize_input, A, B, C):
    """Gets pinn prediction for the forward problem."""
    if normalize_input:
        t_norm = phf.normalize_time(t_eval.reshape(-1,1), t_min, t_max)
    else:
        t_norm = t_eval.reshape(-1,1)
    t_plot = tf.convert_to_tensor(t_norm, dtype=tf.float32)
    loss_phys = phf.physics_loss(model, t_plot, A, B, C, t_min, t_max, normalize_input)

    return (model(t_plot).numpy(), loss_phys)

def plot_results(t_eval, sol, y_pinn):
    """Plots PINN prediction for forward problem."""
    #NOTE this function not needed for jupyter notebook, just in case if pinn.py is desired to be ran directly
    plt.figure(figsize=(12,8))
    labels = ['x','y','z']
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t_eval, sol.y[i],  'k-', label="RK45 (Reference)")
        plt.plot(t_eval, y_pinn[:,i],'b--', label="PINN")
        plt.ylabel(labels[i])
        if i==0:
            plt.title("Pinn prediction for r=28, IC = (1, 1, 1)")
        if i==2:
            plt.xlabel("t")
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
