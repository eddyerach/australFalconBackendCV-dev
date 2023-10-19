import cv2

class GrapeDetector:
    def __init__(self):
        # Initialize the grape detection model or parameters here
        # You can load a pre-trained model or set up detection parameters

        # For example, if you are using OpenCV's CascadeClassifier:
        self.grape_cascade = cv2.CascadeClassifier("grape_cascade.xml")

    def detect_grapes(self, segmented_image):
        """
        Detect individual grapes within a segmented grape bunch image.

        Args:
            segmented_image (numpy.ndarray): Segmented image of a grape bunch as a NumPy array.

        Returns:
            int: Number of detected grapes.
            float: Total area of all detected grapes.
        """
        try:
            # Convert the segmented image to grayscale for grape detection
            gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

            # Detect grapes using the cascade classifier
            grapes = self.grape_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            # Initialize the total number of grapes and total area
            num_detected_grapes = len(grapes)
            total_grape_area = 0.0

            # Draw rectangles around detected grapes and calculate total area
            for (x, y, w, h) in grapes:
                cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                total_grape_area += w * h

            return num_detected_grapes, total_grape_area

        except Exception as e:
            raise Exception(f"Error detecting grapes: {str(e)}")

# Example usage:
# grape_detector = GrapeDetector()
# num_detected_grapes, total_grape_area = grape_detector.detect_grapes(segmented_bunch_image)
