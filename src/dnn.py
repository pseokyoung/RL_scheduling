from tensorflow import keras
from tensorflow.keras.layers import *

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .utility import *

class DNN():
    
    def __init__(self,
                 df, input_var, output_var, num_layers, num_neurons):
        
        self.df = df.copy()
        self.input_var = input_var
        self.output_var = output_var
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.df_train = None
        self.df_test = None
        self.scaler_x = None
        self.scaler_y = None
        self.dnn = None
        
    def train_test_split(self, test_size=0.3, shuffle=True):
        self.df_train, self.df_test = train_test_split(self.df, test_size = test_size, random_state=42, shuffle=shuffle)
        
    def scaler(self,scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler_x = StandardScaler()
            self.scaler_x.fit(self.df_train[self.input_var])
            self.scaler_y = StandardScaler()
            self.scaler_y.fit(self.df_train[self.output_var])
            
        elif scaler_type == 'minmax':
            self.scaler_x = MinMaxScaler()
            self.scaler_x.fit(self.df_train[self.input_var])
            self.scaler_y = MinMaxScaler()
            self.scaler_y.fit(self.df_train[self.output_var])            
        
    def model(self, activation = 'relu'):
        self.dnn = keras.Sequential()
        self.dnn.add(Dense(self.num_neurons, activation=activation, input_shape=[len(self.input_var)]))
        for i in range(self.num_layers-1):
            self.dnn.add(Dense(self.num_neurons, activation=activation))
        self.dnn.add(Dense(len(self.output_var)))

        optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
        self.dnn.compile(loss='mse',
                    optimizer = optimizer)
        
    def fit(self, epochs=1000, callbacks='early_stopping', verbose=2):
        if callbacks == 'early_stopping':
            callbacks = keras.callbacks.EarlyStopping(patience=30, restore_best_weights= True, monitor='val_loss')
            
        self.history = self.dnn.fit(self.scaler_x.transform(self.df_train[self.input_var]), 
                                    self.scaler_y.transform(self.df_train[self.output_var]),
                                    epochs=epochs, callbacks=callbacks, verbose=verbose,
                                    validation_data =(self.scaler_x.transform(self.df_test[self.input_var]), 
                                                      self.scaler_y.transform(self.df_test[self.output_var])))
        
    def save_model(self, save_path, file_name):
        savefile(self.dnn, save_path, file_name, 'model')
        
        
    