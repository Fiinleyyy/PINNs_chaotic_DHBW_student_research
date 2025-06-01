import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pinn_helper_functions as phf
import helper_functions as hf



# ──────────────── Create train_step with its own tf.function cache ────────────────
def create_train_step():
    # @tf.function decorater is wrapped by a method, otherwise creating more than one model instance won't be allowed by tf
    @tf.function
    def train_step(model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input):
        """
        Performs a single training step: computes losses, gradients, and updates the model weights and Lorenz parameters.
        Returns the total loss, physics loss, and data / initial condition loss.
        """
        with tf.GradientTape() as tape:
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            if data_active:
                loss_data = phf.data_loss(model, t_data, y_data)
                loss = alpha * loss_data + (1 - alpha) * loss_phys
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss, loss_data, loss_phys
            else:
                loss_ic = phf.initial_condition_loss(model, t_initial, initial_conditions, t_min, t_max, normalize_input)
                loss = alpha * loss_ic + (1 - alpha) * loss_phys
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss, loss_ic, loss_phys
    return train_step

# ──────────────── Train function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, data_active, t_data, y_data):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)
    model.optimizer = optimizer_class(learning_rate=lr_schedule)

    train_step = create_train_step()

    print("Training started...")
    for epoch in range(epochs):
        t_collocation = phf.sample_collocation(t_min, t_max, collocation_points, normalize_input)
        step_loss, ic_loss, phy_loss = train_step(model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input)
        if epoch % 1000 == 0:
            if not data_active:
                print(f"Epoch {epoch} Loss: {step_loss} | IC-Loss: {ic_loss} | Physics-Loss: {phy_loss}")
            else:
                print(f"Epoch {epoch} Loss: {step_loss} | Data-Loss: {ic_loss} | Physics-Loss: {phy_loss}")

    print("Training finished!")

def pinn_predict(model, t_eval, t_min, t_max, normalize_input, A, B, C):
    if normalize_input:
        t_norm = phf.normalize_time(t_eval.reshape(-1,1), t_min, t_max)
    else:
        t_norm = t_eval.reshape(-1,1)
    t_plot = tf.convert_to_tensor(t_norm, dtype=tf.float32)

    loss_phys = phf.physics_loss(model, t_plot, A, B, C, t_min, t_max, normalize_input)

    return (model(t_plot).numpy(), loss_phys)

def plot_results(t_eval, sol, y_pinn):
    #NOTE this function not needed for jupyter notebook, just in case if pinn.py is desired to be ran directly
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
#NOTE: not needed for jupyter notebook, remove later on
def main():
    # System Parameter
    A = 1
    B = 1
    C = 1
    INITIAL_CONDITIONS = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Time domain
    t_min, t_max = 0.0, 10.0

    # PINN Architektur
    HIDDEN_LAYER = 6
    NEURONS_PER_LAYER = 30
    ACTIVATION_FUNCTION = tf.keras.activations.silu
    WEIGHT_INITIALIZATION = tf.keras.initializers.GlorotUniform

    # Trainings-Hyperparameter
    LEARNING_RATE = 0.01
    DECAY_RATE = 0.09
    OPTIMIZER = tf.keras.optimizers.Adam
    EPOCHS = 5000
    COLLOCATION_POINTS = 1024
    ALPHA_DATA = 0.5
    NORMALIZE_INPUT = False
    DATA_ACTIVE = True

    # Create reference solution
    t_eval, sol = hf.ref_solution(A, B, C, t_min, t_max, INITIAL_CONDITIONS)

    # Add noise to reference solution
    t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max)

    if NORMALIZE_INPUT:
        t_data = phf.normalize_time(t_data, t_min, t_max)

    # Modell bauen
    model = phf.build_pinn_network(HIDDEN_LAYER, NEURONS_PER_LAYER, ACTIVATION_FUNCTION, WEIGHT_INITIALIZATION)

    # Trainieren
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
    main()
