# utils.py

import os

# Constants
CONFIG_PATH = os.path.expanduser("~/.grape_count_config")  # Example configuration file path

# Utility Functions
def save_config(config_data):
    """
    Save configuration data to a configuration file.

    Args:
        config_data (dict): Dictionary containing configuration data.
    """
    try:
        with open(CONFIG_PATH, 'w') as config_file:
            for key, value in config_data.items():
                config_file.write(f"{key}: {value}\n")
    except Exception as e:
        raise Exception(f"Error saving configuration: {str(e)}")

def load_config():
    """
    Load configuration data from a configuration file.

    Returns:
        dict: Dictionary containing loaded configuration data.
    """
    try:
        config_data = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as config_file:
                lines = config_file.readlines()
                for line in lines:
                    key, value = line.strip().split(': ')
                    config_data[key] = value
        return config_data
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def display_image(image, title="Image"):
    """
    Display an image using a GUI window.

    Args:
        image (numpy.ndarray): Image to be displayed as a NumPy array.
        title (str): Title of the image window (default is "Image").
    """
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        raise Exception(f"Error displaying image: {str(e)}")

# Additional utility functions can be added as needed.
