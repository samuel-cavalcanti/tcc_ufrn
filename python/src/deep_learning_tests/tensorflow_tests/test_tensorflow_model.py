import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf


# class NewLayer(tf.keras.layers.Layer):


#     def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)

#         self.alpha = tf.keras.layers.Dense(2)
#         self.cos_layer= tf.keras.layers.Dense(2,use_bias=False)
#         self.sin_layer= tf.keras.layers.Dense(2,use_bias=False)

    
#     # def call(self, theta, *args, **kwargs):

#     #     alpha= self.al

#     #     return super().call(inputs, *args, **kwargs)



def build_model_regression(input_shape: tuple, n_nodes: int,output_nodes:int) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_nodes, activation=tf.keras.activations.relu, input_shape=input_shape))
    # model.add(tf.keras.layers.Dense(n_nodes, activation=tf.keras.activations.relu))
    # model.add(tf.keras.layers.Dense(n_nodes, activation=tf.keras.activations.relu))

    # model.add(tf.keras.layers.Dense(n_nodes, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(output_nodes))
    model.compile(optimizer="rmsprop", loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.mae])

    return model

def main() -> None:
  
    with open('dataset_test.npy', 'rb') as f:
        x = np.load(f).astype(np.float32)
        y = np.load(f).astype(np.float32)

    index= np.arange(len(x))

    np.random.shuffle(index)

    x =x[index]
    y = y[index]

 

    model = build_model_regression((1,),20,y.shape[1])



    history = model.fit(x,y,validation_split=0.4,epochs=100).history

    pred= model.predict(x)
    print(pred.shape)

    print(history.keys())
    
    time_in_epochs = np.arange(len(history['loss']))*10

    plt.figure(0)
    plt.plot(time_in_epochs, history['loss'], label='TRAIN MSE')
    plt.plot(time_in_epochs, history['val_loss'], 'r-', label='TEST MSE')
    plt.legend()
    plt.figure(1)
    plt.plot(time_in_epochs, history['mean_absolute_error'], label='TRAIN MAE')
    plt.plot(time_in_epochs, history['val_mean_absolute_error'], 'r-', label='TEST MAE')
    plt.legend()

    plt.figure(2)
    plt.scatter(y[:, 0], y[:, 1],label='True')
    plt.scatter(pred[:, 0], pred[:, 1], s=0.5, c='r',label='Predict')
    plt.show()


if __name__ == '__main__':
    main()
