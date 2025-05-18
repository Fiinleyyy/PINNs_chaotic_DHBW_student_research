import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# ──────────────── Global configuration ────────────────
# Lorenz system parameter
A = 0.9
B = 1.5
C = 1.7

#A = 10
#B = 28
#C = 8/3
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
    dxdt = A * (y - x)
    dydt = x * (B - z) - y
    dzdt = x * y - C * z
    return dxdt, dydt, dzdt


# ──────────────── Build network ────────────────
def build_network():
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


# ──────────────── PINN functions ────────────────
# NOTE: Add normalization functions

def sample_collocation():
    t_collocation = t_min + (t_max - t_min) * np.random.rand(COLLOCATION_POINTS)
    t_collocation = np.expand_dims(t_collocation, axis=1)
    t_collocation = tf.convert_to_tensor(t_collocation, dtype=tf.float32)
    return t_collocation

def y_pred_function(model, t_collocation):
    with tf.GradientTape() as t:
        t.watch(t_collocation)
        y_pred = model(t_collocation)
    return y_pred


# ──────────────── Build network ────────────────
def physics_loss(model, t_collocation):
    # # initialize GradientTape for automatic differentiation
    # with tf.GradientTape(persistent=True) as t:
    #     t.watch(t_collocation) # derivatives for all time points
    #     pred = model(t_collocation) # prediction of x,y,z for all time points

    # # calculate "true" derivatives by inserting predictions into lorenz system equations
    # jac_matrix = t.batch_jacobian(pred, t_collocation)
    # derivatives_pred = tf.squeeze(jac_matrix, axis=2)  

    # x_pred, y_pred, z_pred = tf.unstack(pred, axis=1)
    # dxdt_pred, dydt_pred, dzdt_pred = tf.unstack(derivatives_pred, axis=1)
    # dxdt_true, dydt_true, dzdt_true = lorenz_system(x_pred, y_pred, z_pred)

    # # Residuen
    # physics_loss = tf.reduce_mean((dxdt_true - dxdt_pred)**2 + (dydt_true - dydt_pred)**2 + (dzdt_true - dzdt_pred)**2)
    # return physics_loss
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

def initial_condition_loss(model, t_initial, initial_conditions):
    t_initial_tensor = tf.constant([[t_initial]], dtype = tf.float32)
    initial_conditions_pred = model(t_initial_tensor)

    true_initial_conditions = tf.reshape(initial_conditions, (1, -1))

    initial_condition_loss = tf.reduce_mean((true_initial_conditions - initial_conditions_pred)**2)
    return initial_condition_loss


# ──────────────── Training step ────────────────
@tf.function # NOTE: maybe use model.trainable_variables instead of model.weights
def train_step(model, t_initial, initial_conditions, t_collocation, alpha):
    with tf.GradientTape() as t: # NOTE: persistent=true if you want e.g. adaptive weighing of alpha -> reference Sophie Steger code
        loss_ic = initial_condition_loss(model, t_initial, initial_conditions)
        loss_phys = physics_loss(model, t_collocation)

        loss = alpha * loss_ic + (1 - alpha) * loss_phys
    
    grads = t.gradient(loss, model.weights)
    model.optimizer.apply_gradients(zip(grads, model.weights))
    return loss, loss_ic, loss_phys # NOTE: maybe return phy+ic loss seperately for analysis

def train(model, t_initial, initial_conditions):
         
        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=1000,
            decay_rate=DECAY_RATE)  
                                
        # Adam optimizer with default settings for momentum
        model.optimizer = OPTIMIZER(learning_rate=lr_schedule) 

        print("Training started...")
        for epoch in range(EPOCHS):
            # DEBUG
            # if epoch < 2000:
            #     alpha = 1.0
            # else:
            #     alpha = ALPHA_DATA
            t_collocation = sample_collocation() 

            # perform one train step
            step_loss, ic_loss, phy_loss = train_step(model, t_initial, initial_conditions, t_collocation, ALPHA_DATA)

            if epoch % 1000 == 0:
                print("Epoch", epoch, "Loss", step_loss, f"IC-Loss: {ic_loss} Physics_loss: {phy_loss}")
                                
        print("Training finished!")


def ref_solution():
    """
    Calculates the reference solution of the Lorenz system using solve_ivp (RK45).
    Returns the time points and the solution.
    """
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
        INITIAL_CONDITIONS,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9
    )
    return t_eval, sol

def pinn_predict(model, t_eval):
    """
    Applies the trained PINN to the time points t_eval and returns the prediction.
    """
    t_plot = tf.convert_to_tensor(t_eval.reshape(-1,1), dtype=tf.float32)
    y_pinn = model(t_plot).numpy()
    return y_pinn

def plot_results(t_eval, sol, y_pinn):
    plt.figure(figsize=(12,8)) 
    labels = ['x','y','z']
    for i in range(3):
        plt.subplot(3,1,i+1)
        # Plot reference solution (RK45)
        plt.plot(t_eval, sol.y[i],  'k-', label="RK45 (Reference)")
        # Plot PINN prediction
        plt.plot(t_eval, y_pinn[:,i],'r--', label="PINN")
        plt.ylabel(labels[i])
        if i==0:
            # Title only in the first subplot
            plt.title("Comparison PINN vs. RK45 for the Lorenz system")
        if i==2:
            # x-axis label only in the last subplot
            plt.xlabel("t")
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    # 1) Build network
    model = build_network()

    # 2) Calculate reference solution
    t_eval, sol = ref_solution()

    # 3) Train PINN
    train(model, t_initial=t_min, initial_conditions=INITIAL_CONDITIONS)

    # 4) Query PINN prediction
    y_pinn = pinn_predict(model, t_eval)

    # 5) Plot: RK45 vs PINN
    plot_results(t_eval, sol, y_pinn)

    print("Example PINN output:")
    print("##########################")
    print(y_pinn[:5])

if __name__ == "__main__":
    main()