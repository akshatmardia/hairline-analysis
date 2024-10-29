import cv2
import numpy as np
from preprocess import preprocess_image

def detect_hairline(image: np.ndarray) -> np.ndarray:
    # Canny edge detection
    lower_threshold = 50
    upper_threshold = 150

    preprocessed = preprocess_image(image)
    edges = cv2.Canny(preprocessed, lower_threshold, upper_threshold)
    
    # Use morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges