import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pinn_helper_functions as phf
import helper_functions as hf

def create_train_step():
    # @tf.function decorater is wrapped by a method, otherwise creating more than one model instance won't be allowed by tf
    @tf.function
    def train_step(model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input, chaotic):
        """
        Performs a single training step: computes losses, gradients, and updates the model weights and Lorenz parameters.
        Returns the total loss, physics loss, and data / initial condition loss.
        """
        with tf.GradientTape() as tape:
            # Berechne Physics-Loss und skaliere sie
            loss_phys = phf.physics_loss(model, t_collocation, A, B, C, t_min, t_max, normalize_input)
            
            if chaotic:
                loss_phys = loss_phys / tf.reduce_mean(loss_phys + 1e-8)  # Normalisierung

            if data_active:
                # Berechne Data-Loss
                loss_data = phf.data_loss(model, t_data, y_data)
                # Kombiniere die Verluste mit dynamischem Alpha
                loss = alpha * loss_data + (1 - alpha) * loss_phys
            else:
                # Berechne Initial Condition Loss
                loss_ic = phf.initial_condition_loss(model, t_initial, initial_conditions, t_min, t_max, normalize_input)
                loss = alpha * loss_ic + (1 - alpha) * loss_phys

            # Berechne Gradienten und wende Gradient Clipping an
            grads = tape.gradient(loss, model.trainable_variables)
            #clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]  # Clipping
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Rückgabe der Verluste
            if data_active:
                return loss, loss_data, loss_phys
            else:
                return loss, loss_ic, loss_phys
    return train_step

# ──────────────── Train function ────────────────
def train(model, t_initial, initial_conditions, A, B, C, t_min, t_max, collocation_points, alpha, learning_rate, decay_rate, epochs, optimizer_class, normalize_input, data_active, t_data, y_data, chaotic=False):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)
    model.optimizer = optimizer_class(learning_rate=lr_schedule)

    # Verwende die train_step-Funktion aus create_train_step
    train_step = create_train_step()

    print("Training started...")

    for epoch in range(epochs):
        # Dynamische Anpassung von Alpha
        #dynamic_alpha = max(0.1, alpha * (1 - epoch / epochs))  # Reduziere Alpha über die Zeit

        # Collocation-Punkte neu sampeln
        t_collocation = phf.sample_collocation(t_min, t_max, collocation_points, normalize_input)

        # Führe einen Trainingsschritt aus
        step_loss, ic_or_data_loss, phy_loss = train_step(
            model, t_initial, initial_conditions, t_collocation, alpha, A, B, C, t_min, t_max, data_active, t_data, y_data, normalize_input, chaotic
        )

        # Ausgabe alle 100 Epochen
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Loss: {step_loss:.4e} | Data/IC-Loss: {ic_or_data_loss:.4e} | Physics-Loss: {phy_loss:.4e}")

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
        plt.plot(t_eval, y_pinn[:,i],'b--', label="PINN")
        plt.ylabel(labels[i])
        if i==0:
            plt.title("Pinn prediction for r=28, IC = (1, 1, 1)")
        if i==2:
            plt.xlabel("t")
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# ──────────────── Main Routine ──────────────
#NOTE: not needed for jupyter notebook, remove later on
def main():
    # System Parameter
    A, B, C = 10, 0.5, 8/3  # Lorenz system parameters
    INITIAL_CONDITIONS = np.array([1, 1, 1], dtype=np.float32)

    # Time domain
    t_min, t_max = 0.0, 15.0

    # PINN Architektur
    HIDDEN_LAYER = 6
    NEURONS_PER_LAYER = 30  # Erhöhe die Anzahl der Neuronen
    ACTIVATION_FUNCTION = tf.keras.activations.silu
    WEIGHT_INITIALIZATION = tf.keras.initializers.GlorotUniform  # Verwende He-Normal-Initialisierung

    # Trainings-Hyperparameter
    LEARNING_RATE = 0.01  # Reduziere die Anfangslernrate
    DECAY_RATE = 0.09  # Verlangsamt die Abnahme der Lernrate
    OPTIMIZER = tf.keras.optimizers.Adam
    EPOCHS = 5000  # Erhöhe die Anzahl der Epochen
    COLLOCATION_POINTS = 1024  # Erhöhe die Anzahl der Collocation-Punkte
    ALPHA_DATA = 0.5  # Physics-Loss stärker gewichten
    NORMALIZE_INPUT = False
    DATA_ACTIVE = False

    # Create reference solution
    t_eval, sol = hf.ref_solution(A, B, C, t_min, t_max, INITIAL_CONDITIONS)

    # Add noise to reference solution
    t_data, y_data = hf.generate_noisy_data(sol, t_min, t_max, noise_factor=0.01)

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
        y_data=y_data,
        chaotic=False
    )

    # PINN prediction
    y_pinn, phy_loss = pinn_predict(model, t_eval, t_min, t_max, normalize_input=NORMALIZE_INPUT, A=A, B=B, C=C)

    # Plot
    plot_results(t_eval, sol, y_pinn)

if __name__ == "__main__":
    main()