import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, Concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
print(tf.config.list_physical_devices())
print(tf.__version__)

# Set the directory paths
with_reflection_dir = 'images_with_reflection'
without_reflection_dir = 'images_with_reflection'

# Image size (adjust as needed)
IMG_SIZE = (256, 256)

# Load and preprocess images
def load_image(img_path, img_size=(256, 256)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1)  # Force single channel (grayscale)
    img = tf.image.resize(img, img_size)  # Resize to consistent shape
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    return img

# Load paired images
def load_paired_images(with_reflection_path, without_reflection_path):
    with_reflection_img = load_image(with_reflection_path)
    without_reflection_img = load_image(without_reflection_path)
    return with_reflection_img, without_reflection_img

# Get the list of file paths for paired images
with_reflection_paths = sorted([os.path.join(with_reflection_dir, fname) for fname in os.listdir(with_reflection_dir)])
without_reflection_paths = sorted([os.path.join(without_reflection_dir, fname) for fname in os.listdir(without_reflection_dir)])

# Create TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((with_reflection_paths, without_reflection_paths))
dataset = dataset.map(lambda x, y: load_paired_images(x, y), num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=100).prefetch(tf.data.AUTOTUNE)

# Determine the size of the dataset
total_size = len(with_reflection_paths)
val_size = int(0.2 * total_size)  # 10% for validation
train_size = total_size - val_size

# Split the dataset into training and validation
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

def build_reflection_removal_model(input_shape=(256, 256, 1)):
    model = Sequential()

    # Encoder part
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Replace UpSampling2D with Conv2DTranspose for learnable upsampling
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))  # Downsample back to original size

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))  # Downsample back to original size

    # Output Layer to ensure correct output shape
    model.add(Conv2D(1, (3, 3), padding='same', activation='sigmoid'))

    return model

# Loss Function and Compilation
def compile_model():
    model = build_reflection_removal_model()
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['accuracy', 'mae'])
    return model

# Train the Model
def train_model(model, train_dataset, val_dataset):
    model.fit(train_dataset, validation_data=val_dataset, epochs=30, batch_size=128)

# Build, compile and train model
model = compile_model()
train_model(model, train_dataset, val_dataset)
# Save the entire model as a `.keras` zip archive.
model.save('cnn_model.keras')
#model.summary()
