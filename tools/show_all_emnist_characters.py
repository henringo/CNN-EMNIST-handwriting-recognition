#!/usr/bin/env python3
"""
Show original EMNIST training pictures for every character (A-Z, 0-9).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ml.model import class_labels

def show_all_emnist_characters():
    """Show EMNIST samples for all characters."""
    print("Loading EMNIST dataset...")
    
    # Load EMNIST byclass
    emnist = datasets.EMNIST(
        root='data',
        split='byclass',
        train=True,  # Use training set for more variety
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"EMNIST dataset loaded: {len(emnist)} samples")
    
    # Get class labels
    class_labels_list = class_labels()
    print(f"Classes: {class_labels_list}")
    
    # Create mapping from our class labels to EMNIST classes
    # EMNIST byclass: 0-9 are 0-9, A-Z are 10-35
    emnist_class_mapping = {}
    for i, char in enumerate(class_labels_list):
        if i < 26:  # Letters A-Z
            emnist_class = i + 10  # A=10, B=11, etc.
        else:  # Digits 0-9
            emnist_class = i - 26  # 0=0, 1=1, etc.
        emnist_class_mapping[char] = emnist_class
    
    print("Creating character gallery...")
    
    # Create a large figure to show all characters
    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, char in enumerate(class_labels_list):
        emnist_class = emnist_class_mapping[char]
        
        # Find samples of this character
        samples = []
        for j, label in enumerate(emnist.targets):
            if label == emnist_class:
                samples.append(j)
                if len(samples) >= 5:  # Get 5 samples
                    break
        
        if samples:
            # Pick a random sample
            sample_idx = samples[0]  # Use first sample for consistency
            img, _ = emnist[sample_idx]
            img_np = img.squeeze().numpy()
            
            # Show the character
            axes[i].imshow(img_np, cmap='gray')
            axes[i].set_title(f'{char}', fontsize=16, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No {char}', ha='center', va='center', fontsize=12)
            axes[i].axis('off')
    
    plt.suptitle('EMNIST Training Characters (A-Z, 0-9)', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('emnist_all_characters.png', dpi=150, bbox_inches='tight')
    print("Saved all characters to emnist_all_characters.png")
    
    # Create detailed view for each character with multiple samples
    print("Creating detailed character views...")
    
    for char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
        emnist_class = emnist_class_mapping[char]
        
        # Find samples of this character
        samples = []
        for j, label in enumerate(emnist.targets):
            if label == emnist_class:
                samples.append(j)
                if len(samples) >= 12:  # Get 12 samples
                    break
        
        if samples:
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()
            
            for i, sample_idx in enumerate(samples):
                img, _ = emnist[sample_idx]
                img_np = img.squeeze().numpy()
                
                axes[i].imshow(img_np, cmap='gray')
                axes[i].set_title(f'{char} Sample {i+1}')
                axes[i].axis('off')
            
            plt.suptitle(f'EMNIST {char} Samples (Training Data)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'emnist_{char.lower()}_samples.png', dpi=150, bbox_inches='tight')
            print(f"Saved {char} samples to emnist_{char.lower()}_samples.png")
    
    # Show digits too
    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        emnist_class = emnist_class_mapping[digit]
        
        # Find samples of this digit
        samples = []
        for j, label in enumerate(emnist.targets):
            if label == emnist_class:
                samples.append(j)
                if len(samples) >= 12:  # Get 12 samples
                    break
        
        if samples:
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()
            
            for i, sample_idx in enumerate(samples):
                img, _ = emnist[sample_idx]
                img_np = img.squeeze().numpy()
                
                axes[i].imshow(img_np, cmap='gray')
                axes[i].set_title(f'{digit} Sample {i+1}')
                axes[i].axis('off')
            
            plt.suptitle(f'EMNIST {digit} Samples (Training Data)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'emnist_{digit}_samples.png', dpi=150, bbox_inches='tight')
            print(f"Saved {digit} samples to emnist_{digit}_samples.png")
    
    print("\n=== Summary ===")
    print("Generated images:")
    print("- emnist_all_characters.png: Overview of all 36 characters")
    print("- emnist_[letter]_samples.png: 12 samples each for A-Z")
    print("- emnist_[digit]_samples.png: 12 samples each for 0-9")
    print("\nThese show exactly what the model was trained on!")

if __name__ == "__main__":
    print("=== EMNIST Character Gallery ===")
    show_all_emnist_characters()
    print("\nThis shows the original EMNIST training data for every character!")
