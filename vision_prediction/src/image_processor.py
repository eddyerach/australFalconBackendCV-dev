import cv2

class ImageProcessor:
    def __init__(self):
        pass

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess an image.

        Args:
            image_path (str): Path to the input image.
            target_size (tuple): Size to which the image should be resized.

        Returns:
            numpy.ndarray: Preprocessed image as a NumPy array.
        """
        try:
            # Load the image using OpenCV
            image = cv2.imread(image_path)

            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # Resize the image to the target size
            image = cv2.resize(image, target_size)

            # Convert the image to a format suitable for computer vision tasks (e.g., BGR to RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Normalize pixel values to the range [0, 1]
            image = image / 255.0

            return image

        except Exception as e:
            raise Exception(f"Error loading and preprocessing image: {str(e)}")

# Example usage:
# image_processor = ImageProcessor()
# preprocessed_image = image_processor.load_and_preprocess_image("input.jpg")
