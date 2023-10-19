import cv2

class GrapeBunchDetector:
    def __init__(self):
        # Initialize the grape bunch detection model or parameters here
        # You can load a pre-trained model or set up detection parameters

        # For example, if you are using OpenCV's CascadeClassifier:
        self.bunch_cascade = cv2.CascadeClassifier("grape_bunch_cascade.xml")

    def detect_and_segment_bunch(self, image):
        """
        Detect and segment grape bunches in an image.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: Segmented image containing detected grape bunches.
            float: Total area of the detected grape bunches.
        """
        try:
            # Convert the image to grayscale for cascade detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect grape bunches using the cascade classifier
            bunches = self.bunch_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            # Initialize the segmented image
            segmented_image = image.copy()

            # Initialize total area
            total_area = 0.0

            # Draw rectangles around detected bunches and calculate total area
            for (x, y, w, h) in bunches:
                cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                total_area += w * h

            return segmented_image, total_area

        except Exception as e:
            raise Exception(f"Error detecting and segmenting grape bunches: {str(e)}")

# Example usage:
# grape_bunch_detector = GrapeBunchDetector()
# segmented_image, bunch_area = grape_bunch_detector.detect_and_segment_bunch(input_image)
