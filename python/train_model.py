from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from main import ENVIRONMENT_PATH, KINEMATIC_PATH, Array32


def load_kinematic_dataset() -> tuple[np.ndarray, np.ndarray]:
    with open('memory.npy', 'rb') as f:
        x = np.load(f)
        y = np.load(f)

    return x, y


def load_simulation_dataset() -> Array32:
    with open('simulation.npy', 'rb') as f:
        s = np.load(f)

    return s


def load_kinematic_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, "elu"),
        tf.keras.layers.Dense(4, "elu"),
        tf.keras.layers.Dense(2),
    ])

    model.build((1, 4))

    if Path(KINEMATIC_PATH).exists():
        model.load_weights(KINEMATIC_PATH)

    else:
        print('Unable to load kinematic model !!')
    return model


def load_env_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, "elu"),
        tf.keras.layers.Dense(8, "elu"),
        tf.keras.layers.Dense(4, "elu"),
        tf.keras.layers.Dense(3),
    ])

    model.build((1, 3))

    if Path(ENVIRONMENT_PATH).exists():
        model.load_weights(ENVIRONMENT_PATH)

    else:
        print('Unable to load ENV model !!')
    return model


def train_kinematic_model():
    x, y = load_kinematic_dataset()

    model = load_kinematic_model()

    history = train_model(model, x, y, KINEMATIC_PATH)

    plot_history(history)


def train_env_model():
    sim_data = load_simulation_dataset()
    x, y = preprocessing_sim_data(sim_data)


    model = load_env_model()
    history = train_model(model, x, y, ENVIRONMENT_PATH)
    plot_history(history)


def train_model(model: tf.keras.Model, x: Array32, y: Array32, weights_path: str) -> dict[str, list[float]]:
    model.compile(tf.keras.optimizers.RMSprop(0.000025),  # 0.00025)  # 025
                  metrics=["mae"],
                  loss=tf.keras.losses.mse)

    print('x shape: ', x.shape)
    print('y shape: ', y.shape)

    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    pred_size = int(x.shape[0]*0.60)

    x_pred = x[:pred_size]
    y_pred = y[:pred_size]

    x_test = x[pred_size:]
    y_test = y[pred_size:]

    print('x pred shape', x_pred.shape)

    val = model.evaluate(x_test, y_test)

    history = model.fit(x_pred, y_pred,
                        # batch_size=64,
                        validation_split=0.2,
                        epochs=700,
                        ).history
    val = model.evaluate(x_test, y_test)
    # Loss MSE: 0.03300042450428009 MAE 0.04207644611597061 
    print(f'{weights_path} Loss MSE: {val[0]} MAE {val[1]} ')

    model.save_weights(weights_path, save_format='h5')

    return history


def preprocessing_sim_data(sim_data: Array32) -> tuple[Array32, Array32]:
    input_index = np.arange(0, len(sim_data), 2)
    output_index = np.arange(1, len(sim_data), 2)

    sim_input = sim_data[input_index]
    sim_output = sim_data[output_index]

    t_index = 0
    sx = 1
    sy = 2
    theta_index = 3
    vel_wheel_l = -2
    vel_wheel_r = -1

    s_1 = sim_input[:, sx:sy + 1]
    theta_1 = sim_input[:, theta_index].reshape((-1, 1))
    vel_wheel = sim_input[:, vel_wheel_l:]

    s_2 = sim_output[:, sx:sy + 1]
    theta_2 = sim_output[:, theta_index].reshape((-1, 1))
    
    delta_s = s_2 - s_1

    delta_theta = theta_2 - theta_1
    delta_t =  sim_input[:,t_index] - sim_output[:,t_index]

    delta_theta = delta_theta.reshape((-1, 1))
    delta_t = delta_t.reshape((-1, 1))

    v = delta_s/delta_t
    omega = delta_theta/delta_t
    
   

    x = np.hstack((vel_wheel,theta_1))
    y =  np.hstack((v,theta_2))

    return x, y


def main():
    # train_kinematic_model()
    train_env_model()


def plot_history(history: dict[str, list[float]]):
    time_in_epochs = list(range(len(history['loss'])))
    plt.figure(0)
    plt.plot(time_in_epochs, history['loss'], label=' mse')
    plt.plot(time_in_epochs, history['val_loss'], 'r-', label=' val mse')
    plt.legend()
    plt.figure(1)
    plt.plot(time_in_epochs, history['mae'], label=' mae')
    plt.plot(time_in_epochs, history['val_mae'], 'r-', label='Val mae')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
