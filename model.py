import tensorflow as tf

keras = tf.keras
from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten, LeakyReLU, ReLU


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Invalid device or cannot modify virtual devices once initialized')


def create_model(input_shape, output_shape):
    input = Input(shape=input_shape)
    layer = Conv2D(8, kernel_size=5, strides=2, activation='relu')(input)
    layer = BatchNormalization()(layer)
    layer = Conv2D(16, kernel_size=3, strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, kernel_size=3, strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, kernel_size=3, strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

    merged = Conv2D(256, kernel_size=3, strides=1, activation='relu', name='final_conv')(layer)
    merged = BatchNormalization()(merged)
    merged = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(merged)

    merged = Flatten()(merged)

    merged = Dense(512, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(8, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(output_shape, activation="softmax", name='predictions')(merged)

    model = Model(inputs=input, outputs=output)
    return model


def load_model() -> keras.Model:
    model = create_model((128, 128, 1), 5)
    print('created model')
    model.load_weights('/hpf/largeprojects/dsingh/cts/results/1539211/3D_CNN_collapsed.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = load_model()
    model.summary()
