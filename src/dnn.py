import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

from sklearn.preprocessing import StandardScaler

def DNN(num_x, num_y, num_layers, num_neurons):
    model = keras.Sequential()
    model.add(  Dense( num_neurons, activation="relu", input_shape=[num_x] )  )
    for i in range(num_layers-1):
        model.add(  Dense( num_neurons, activation="relu" )  )
    model.add(  Dense( num_y )  )

    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse',
                  optimizer = optimizer)
    return model