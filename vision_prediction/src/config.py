# config.py

# Image processing settings
IMAGE_TARGET_SIZE = (224, 224)  # Target size for image resizing
IMAGE_NORMALIZATION = True       # Whether to normalize pixel values

# Paths to model files
GRAPE_BUNCH_MODEL_PATH = "models/grape_bunch_model.h5"
GRAPE_DETECTION_MODEL_PATH = "models/grape_detection_model.h5"

# Deep learning model hyperparameters
DNN_HIDDEN_UNITS = [128, 64]     # Number of hidden units in the DNN layers
DNN_ACTIVATION = 'relu'          # Activation function for DNN layers
DNN_LEARNING_RATE = 0.001        # Learning rate for training the DNN
DNN_EPOCHS = 20                  # Number of training epochs

# File paths
CONFIG_PATH = "config.json"      # Path to configuration file
LOG_PATH = "logs/log.txt"        # Path to log file
DATA_DIR = "data/"               # Directory containing data files

# Other project-specific constants and configurations can be added here
