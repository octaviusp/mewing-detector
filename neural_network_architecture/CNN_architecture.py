import tensorflow as tf

# creating neural net architecture
# Sequential, conv2d 64 filters (remember a filter is like an image transformation for capturing features)
# filters 3x3 size, activation rectifier linear unit.
# inut shape 28x28x1 as resize above.
# MaxPooling2d 2x2, it means grabbing in a submatrix of 2x2 the pixel with most value.
# Flatten means flatten the 28x28 matrix into 1x(28.28) dimension , 1 dimensional array.
# Thereefore normal dense layers (fully conected neurons) with relu
# and the last layer with softamx activation for multiclassification.
#
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 1 # 3 FOR RGB CHANNEL

INPUT_SHAPE = (IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)

FILTER_SIZE = (3,3)
FILTER_QUANTITY = 64
POOL_SIZE = 2

FIRST_DENSE_LAYER_UNITS = 64
SECOND_DENSE_LAYER_UNITS = 1 # FOR BINARY CLASSIFCATION

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(FILTER_QUANTITY, FILTER_SIZE, activation='relu',
                    input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(POOL_SIZE,POOL_SIZE),
        tf.keras.layers.Conv2D(FILTER_QUANTITY, FILTER_SIZE, activation='relu'),
        tf.keras.layers.MaxPooling2D(POOL_SIZE,POOL_SIZE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(FIRST_DENSE_LAYER_UNITS, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),  # Add dropout with a dropout rate of 20%
        tf.keras.layers.Dense(SECOND_DENSE_LAYER_UNITS, activation=tf.nn.sigmoid)
])
