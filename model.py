from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, Flatten, Dense, Activation, AlphaDropout, Reshape, add
from tensorflow.keras.models import Model

class ResidualConv1DModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def residual_stack(self, X, Filters, Seq, max_pool):

        X = Conv1D(Filters, 1, padding='same', name=Seq + "_conv1", kernel_initializer='glorot_uniform')(X)

        X_shortcut = X
        X = Conv1D(Filters, 3, padding='same', activation="relu", name=Seq + "_conv2",
                   kernel_initializer='glorot_uniform')(X)
        X = Conv1D(Filters, 3, padding='same', name=Seq + "_conv3", kernel_initializer='glorot_uniform')(X)
        X = add([X, X_shortcut])
        X = Activation("relu")(X)

        X_shortcut = X
        X = Conv1D(Filters, 3, padding='same', activation="relu", name=Seq + "_conv4",
                   kernel_initializer='glorot_uniform')(X)
        X = Conv1D(Filters, 3, padding='same', name=Seq + "_conv5", kernel_initializer='glorot_uniform')(X)
        X = add([X, X_shortcut])
        X = Activation("relu")(X)

        if max_pool:
            X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
        return X

    def build_model(self):
        X_input = Input(shape=self.input_shape)

        X = self.residual_stack(X_input, 32, "ReStk1", False)  # shape:(1,512,32)
        X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)

        X = self.residual_stack(X, 32, "ReStk2", True)  # shape:(1,256,32)

        X = self.residual_stack(X, 32, "ReStk3", True)  # shape:(1,128,32)

        X = self.residual_stack(X, 32, "ReStk4", True)  # shape:(1,64,32)

        X = self.residual_stack(X, 32, "ReStk5", True)  # shape:(1,32,32)

        X = self.residual_stack(X, 32, "ReStk6", True)  # shape:(1,16,32)

        X = Flatten()(X)
        X = Dense(128, activation='selu', kernel_initializer='he_normal', name="dense1")(X)
        X = AlphaDropout(0.3)(X)

        X = Dense(128, activation='selu', kernel_initializer='he_normal', name="dense2")(X)
        X = AlphaDropout(0.3)(X)

        X = Dense(self.num_classes, kernel_initializer='he_normal', name="dense3")(X)

        X = Activation('softmax')(X)


        model = Model(inputs=X_input, outputs=X)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model



# input_shape = (1024, 2)
# num_classes = 11
# residual_conv1d_model = ResidualConv1DModel(input_shape, num_classes)
# model = residual_conv1d_model.build_model()
# model.summary()
