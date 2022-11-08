import numpy as np
import tensorflow as tf
from matplotlib import pyplot

from datasets import load_data_csv

def convert(simulation_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    up = [0, 0, 0, 1]
    down = [0, 0, 1, 0]
    left = [0, 1, 0, 0]
    right = [1, 0, 0, 0]

    def velocity_to_a(vel: np.ndarray) -> list[int]:

        right_vel, left_vel = vel[0], vel[1]

        match (right_vel, left_vel):

            case (2.0, 0.0):
                return left
            case (0.0, 2.0):
                return right
            case (2.0, 2.0):
                return up
            case (-2, 0, -2.0):
                return down

            case _:
                raise Exception(f'Unable to convert vel: {vel}')

    y: list[list[int]] = list()
    x: list[np.ndarray] = list()

    for line in simulation_data:

        velocity = line[-2:]
        pos = line[:-2]
        vel_d = velocity_to_a(velocity)

        x.append(pos)
        y.append(vel_d)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    """
        Doesn't work :(
    """
    simulation_data =  load_data_csv()

    print(simulation_data.shape)

    x, y = convert(simulation_data)

    y = y.reshape((-1))

    print('x shape', x.shape, 'y shape', y.shape)

    model = tf.keras.models.Sequential([
      
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        x, y, epochs=60, validation_split=0.1, batch_size=4).history

    model.save('model.tf')
    plot_history(history)


def plot_history(history: dict[str, list[float]]) -> None:

    val_loss = history['val_loss']
    loss = history['loss']

    time_in_epochs = list(range(1, len(loss)+1))

    pyplot.title('Effect of insufficient  model capacity on validade loss')
    pyplot.plot(time_in_epochs, loss, label='loss')
    pyplot.plot(time_in_epochs, val_loss, 'b--', label="val_loss")
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Loss')
    pyplot.xticks(time_in_epochs)
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
