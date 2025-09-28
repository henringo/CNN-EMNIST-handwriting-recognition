#!/usr/bin/env python3
"""
Simplified canvas app with minimal preprocessing to match EMNIST better.
"""

import argparse
import os
import sys
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
from tkinter import ttk

import torch

# Ensure project root is importable when running as a script (tools/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.model import class_labels

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def simple_preprocess_image(image_array: np.ndarray) -> torch.Tensor:
    """
    Simplified preprocessing that's closer to EMNIST format.
    """
    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)
    
    # Find bounding box
    coords = np.column_stack(np.where(image_array > 0))
    if len(coords) == 0:
        return torch.zeros((1, 1, 28, 28))
    
    rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
    cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
    
    # Add some padding
    padding = 2
    rmin = max(0, rmin - padding)
    rmax = min(image_array.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(image_array.shape[1] - 1, cmax + padding)
    
    cropped = image_array[rmin:rmax + 1, cmin:cmax + 1]
    
    # Invert (EMNIST has white background, black text)
    inverted = 255 - cropped
    
    # Resize to 28x28 using PIL for better quality
    pil_img = Image.fromarray(inverted)
    resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
    resized_array = np.array(resized, dtype=np.float32)
    
    # Normalize to [0, 1]
    resized_array /= 255.0
    
    # Apply MNIST normalization
    normalized = (resized_array - MNIST_MEAN) / MNIST_STD
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


class SimpleCanvasApp:
    def __init__(self, model_path: str) -> None:
        self.root = tk.Tk()
        self.root.title("Handwriting Recognition (A-Z, 0-9) - Simple")
        self.root.geometry("800x600")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        print("Model loaded successfully!")
        
        # Get class labels
        self.class_labels = class_labels()
        print(f"Classes: {self.class_labels}")
        
        # Canvas setup
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg='white', 
            cursor='crosshair'
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Bind drawing events
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.start_draw)
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=10)
        
        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict, bg='lightblue')
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Prediction display
        self.prediction_frame = tk.Frame(self.root)
        self.prediction_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.prediction_label = tk.Label(self.prediction_frame, text="Prediction: ", font=('Arial', 16))
        self.prediction_label.pack()
        
        self.confidence_label = tk.Label(self.prediction_frame, text="", font=('Arial', 12))
        self.confidence_label.pack()
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        self.drawing = False
        
    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        if self.drawing and self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE
            )
        self.last_x = event.x
        self.last_y = event.y
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Prediction: ")
        self.confidence_label.config(text="")
        
    def predict(self):
        # Get canvas content as image
        canvas_image = self.get_canvas_image()
        
        # Preprocess
        processed = simple_preprocess_image(canvas_image)
        
        # Predict
        with torch.no_grad():
            output = self.model(processed)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        
        # Display results
        pred_letter = self.class_labels[pred_class]
        self.prediction_label.config(text=f"Prediction: {pred_letter}")
        
        # Create top 3 display
        top3_text = "Top 3: "
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            letter = self.class_labels[idx.item()]
            top3_text += f"{letter}({prob.item():.0%})"
            if i < 2:
                top3_text += ", "
        
        self.confidence_label.config(text=top3_text)
        
        # Print detailed results
        print(f"\nPrediction: {pred_letter} (confidence: {confidence:.1%})")
        print("Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            letter = self.class_labels[idx.item()]
            print(f"  {i+1}. {letter}: {prob.item():.1%}")
            
    def get_canvas_image(self) -> np.ndarray:
        """Get canvas content as numpy array using a simpler approach."""
        # Create a PIL image from canvas
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Get all canvas items and redraw them
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'line':
                coords = self.canvas.coords(item)
                if len(coords) >= 4:
                    # Draw line on PIL image
                    draw.line(coords, fill='black', width=8)
        
        # Convert to grayscale and numpy array
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        return img_array
        
    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Simple Handwriting Recognition App')
    parser.add_argument('--model', default='models/emnist36_balanced.pt', help='Path to model file')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return
        
    app = SimpleCanvasApp(args.model)
    app.run()


if __name__ == "__main__":
    main()
