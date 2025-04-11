#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 21:29:21 2024

@author: soumyashekhar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class RGBThermalDetector:
    def __init__(self, temp_threshold=3):
        self.temp_threshold = temp_threshold
        
    def load_thermal_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        # Convert BGR to HSV for better temperature representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Use value channel as temperature proxy (brighter = hotter)
        temp_map = hsv[:,:,2].astype(np.float32)
        
        # Normalize to temperature range (-20 to 120°C)
        temp_map = -20 + (temp_map / 255.0) * 140
        return temp_map, image

    def detect_effluents(self, temp_map):
        # Multi-scale gradient analysis
        gradients = []
        for ksize in [3, 5, 7]:
            dx = cv2.Sobel(temp_map, cv2.CV_32F, 1, 0, ksize=ksize)
            dy = cv2.Sobel(temp_map, cv2.CV_32F, 0, 1, ksize=ksize)
            gradients.append(np.sqrt(dx*dx + dy*dy))
            
        combined_gradient = np.mean(gradients, axis=0)
        
        # Temperature anomaly detection using local statistics
        temp_std = temp_map - cv2.GaussianBlur(temp_map, (31,31), 0)
        temp_anomalies = temp_std > np.percentile(temp_std, 90)
        
        # Combine evidence
        anomaly_score = combined_gradient * temp_anomalies
        
        # Thresholding and morphology
        norm_score = cv2.normalize(anomaly_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm_score, 0, 255, cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find and analyze contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        effluents = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            region = temp_map[y:y+h, x:x+w]
            
            # Calculate temperature statistics
            mean_temp = np.mean(region)
            bg_temp = np.median(temp_map[max(0,y-h):y+2*h, max(0,x-w):x+2*w])
            temp_diff = mean_temp - bg_temp
            
            if temp_diff > self.temp_threshold:
                effluents.append({
                    'bbox': (x, y, w, h),
                    'temperature': mean_temp,
                    'temp_difference': temp_diff,
                    'contour': contour
                })
        
        return effluents, anomaly_score, mask

    def visualize_results(self, original_image, temp_map, anomaly_score, mask, effluents):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        axes[0,0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original RGB Image')
        
        axes[0,1].imshow(temp_map, cmap='inferno')
        axes[0,1].set_title('Temperature Map')
        
        axes[1,0].imshow(anomaly_score, cmap='hot')
        axes[1,0].set_title('Anomaly Score')
        
        axes[1,1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        for eff in effluents:
            x, y, w, h = eff['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[1,1].add_patch(rect)
            axes[1,1].text(x, y-5, f"ΔT: {eff['temp_difference']:.1f}°C", 
                          color='white', bbox=dict(facecolor='red', alpha=0.5))
        axes[1,1].set_title('Detected Effluents')
        
        plt.tight_layout()
        return fig

def main():
    detector = RGBThermalDetector(temp_threshold=3)
    image_path = "/Users/soumyashekhar/Desktop/1.jpg"
    
    temp_map, original_image = detector.load_thermal_image(image_path)
    effluents, anomaly_score, mask = detector.detect_effluents(temp_map)
    
    fig = detector.visualize_results(original_image, temp_map, anomaly_score, mask, effluents)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(output_dir / f"thermal_analysis_{timestamp}.png", dpi=300)
    
    for i, eff in enumerate(effluents, 1):
        print(f"\nEffluent {i}:")
        print(f"Temperature Difference: {eff['temp_difference']:.1f}°C")
        print(f"Location: {eff['bbox']}")
        
    plt.show()

if __name__ == "__main__":
    main()