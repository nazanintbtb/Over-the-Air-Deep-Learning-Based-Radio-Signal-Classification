import CNN_LSTM
import load_data as loader
import model as resnet
import tensorflow as tf
import numpy as np
import CNN_model as cnn
import CNN_LSTM as cnn_lstm

input_shape = (1024, 2)
num_classes = 8
# conv1d_model = resnet.ResidualConv1DModel(input_shape, num_classes)
# conv1d_model = cnn.CNN_deep(input_shape, num_classes)
conv1d_model = cnn_lstm.CNN_LSTM(input_shape, num_classes)

model = conv1d_model.build_model()
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="./test.h5", verbose=1,
                                                  save_best_only=True)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=1)

model.fit(loader.get_train_data(64), steps_per_epoch=int(np.ceil(5440/ 64)), epochs=300,
          validation_data=loader.get_valid_data(64), validation_steps=int(np.ceil(1286/ 64)),
          callbacks=[checkpointer, earlystopper])

