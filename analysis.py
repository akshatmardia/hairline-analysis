from detection import detect_hairline
from texture import analyze_texture
import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict

class HairlineAnalyzer:
    def __init__(self, base_path: str):
        
        self.base_path = base_path
        self.results_path = os.path.join(base_path, 'results')
        os.makedirs(self.results_path, exist_ok=True)
    
    def process_weekly_images(self, left_image_path: str, right_image_path: str) -> Dict:
     
        # Read images
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)

        if left_img is None or right_img is None:
            raise ValueError("Could not either or both images")
            
        results = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'left_side': {},
            'right_side': {}
        }
        
        # Process each side
        for side, img in [('left_side', left_img), ('right_side', right_img)]:
            # Detect hairline and analyse texture
            hairline = detect_hairline(img)
            texture_metrics = analyze_texture(img)
            
            # Store results
            results[side] = {
                'hairline_data': hairline,
                'hairline_pixels': np.sum(hairline > 0),
                'texture_metrics': texture_metrics
            }
            
        return results
    
    def visualize_results(self, image: np.ndarray, hairline: np.ndarray, 
                         texture_metrics: Dict, title: str) -> None:
       
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        # Hairline detection
        plt.subplot(132)
        plt.imshow(hairline, cmap='gray')
        plt.title('Hairline Detection')
        
        # Texture metrics
        plt.subplot(133)
        metrics = list(texture_metrics.items())
        plt.bar([m[0] for m in metrics], [m[1] for m in metrics])
        plt.title('Texture Metrics')
        plt.xticks(rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()