from neural_network_architecture import CNN_architecture
from data_preprocessing import x_y_images
from scripts import create_dataset, image_processing
import numpy as np
import tensorflow as tf
import os

checkpoint_path = "training/cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

model = CNN_architecture.model

model.compile(
    optimizer='adam',              # Optimizer
    loss='binary_crossentropy',  # Loss function
    metrics=['accuracy']           # Metric to track
)

x, y = create_dataset.generate()

x_numpy_array = np.array(x)
y_numpy_array = np.array(y)

def train_test_split(array):
    upper_limit = int(len(array)*0.8)
    return array[:upper_limit], array[upper_limit:]

x_train, x_test = train_test_split(x_numpy_array)
y_train, y_test = train_test_split(y_numpy_array)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=12,
    callbacks=[checkpoint_callback]
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

inference_path = "./images/inference_test/"

inference_images = os.listdir(inference_path)

processed_images = []
for inference_image in inference_images:
    processed_image = image_processing.Processor_1.process_on_demand(inference_path+inference_image)
    if processed_image is None:
        print("- Error predicting this image... ", inference_image)
        continue
    processed_images.append(processed_image)

print(np.array(processed_images))
predictions = model.predict(np.array(processed_images))

for index, image in enumerate(inference_images):
    try:
        print(" - Image: ", image, " - Prediction: ", predictions[index])
    except:
        pass
