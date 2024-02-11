from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation,GRU,Dropout,LSTM
from tensorflow.keras.activations import selu, softmax


class CNN_LSTM:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential([
            Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2, strides=2),
            # Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            # MaxPooling1D(pool_size=2, strides=2),
            # Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            # MaxPooling1D(pool_size=2, strides=2),
            Dropout(0.2),
            GRU(128),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(self.num_classes, activation=softmax)
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
