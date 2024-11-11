import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Image Decomposition Function
def decompose_image(image):
    # Convert to grayscale for simplification
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use GaussianBlur to simulate a smooth reflection layer
    reflection_layer = cv2.GaussianBlur(gray, (21, 21), 0)
    # Subtract reflection from original to get estimated clean image
    clean_image = cv2.subtract(gray, reflection_layer)
    # Normalize reflection layer to [0, 1]
    reflection_layer = reflection_layer / 255.0
    gray = gray / 255.0
    return gray, clean_image, reflection_layer

# Load and Preprocess Images
def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))  # Resize for consistency
    gray, clean, reflection = decompose_image(image)
    return gray, clean, reflection, image

# Example Usage
if __name__ == "__main__":
    # Load a sample image
    gray, clean_image, reflection_layer, original_image = load_image('97-2.jpeg')

    # Load the trained model
    model = tf.keras.models.load_model('cnn_model.keras')

    # Predict and display results on the sample
    predicted_clean_image = model.predict(np.expand_dims(np.expand_dims(gray, axis=0), axis=-1))[0]
    
    # Rescale output to [0, 255]
    predicted_clean_image = np.clip(predicted_clean_image * 255, 0, 255).astype(np.uint8).squeeze()
    reflection_layer = np.clip(reflection_layer * 255, 0, 255).astype(np.uint8).squeeze()
    gray = np.clip(gray * 255, 0, 255).astype(np.uint8).squeeze()
    
    # Convert grayscale prediction to 3-channel RGB
    predicted_rgb_image = cv2.cvtColor(predicted_clean_image, cv2.COLOR_GRAY2RGB)

    # Blend with the original color image to retain some color
    alpha = 0.5  # Adjust blending factor as needed
    colored_prediction = cv2.addWeighted(predicted_rgb_image, alpha, original_image, 1 - alpha, 0)

    # Display images using OpenCV
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Reflection Layer", reflection_layer)
    cv2.imshow("Clean Image (Original Decomposed)", clean_image)
    cv2.imshow("Predicted Clean Image (Color Mapped)", colored_prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


