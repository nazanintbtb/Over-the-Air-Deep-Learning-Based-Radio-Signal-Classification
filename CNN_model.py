from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation
from tensorflow.keras.activations import selu, softmax

class CNN_deep:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential([
            Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2, strides=2),
            Flatten(),
            Dense(128, activation=selu),
            Dense(128, activation=selu),
            Dense(self.num_classes, activation=softmax)
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return  model



# input_shape = (1024, 2)
# num_classes = 11

# cnn1d_model = build_cnn1d_model(input_shape, num_classes)

# cnn1d_model.summary()
