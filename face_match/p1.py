import tensorflow as tf

# Check the TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if TensorFlow is using GPU (if you have GPU support)
print("GPU available:" if tf.config.list_physical_devices('GPU') else "No GPU available")