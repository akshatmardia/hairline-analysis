import numpy as np
from skimage.feature import local_binary_pattern
from typing import Dict
from preprocess import preprocess_image

# Using Local Binary Patterns for texture analysis
def analyze_texture(image: np.ndarray) -> Dict:
    preprocessed = preprocess_image(image)

    # LBP parameters
    n_points = 24  # number of points to consider for LBP
    radius = 3     # radius for LBP
    
    # Compute LBP
    lbp = local_binary_pattern(preprocessed, 
                            n_points, 
                            radius, 
                            method='uniform')
    
    # Calculate histogram of LBP values
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), 
                        bins=n_bins, 
                        range=(0, n_bins), 
                        density=True)
    
    # Calculate texture metrics
    metrics = {
        'contrast': np.std(hist),
        'uniformity': np.sum(hist ** 2),
        'entropy': -np.sum(hist * np.log2(hist + 1e-10))
    }
    
    return metrics