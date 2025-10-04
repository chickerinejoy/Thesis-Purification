"""
Example Usage Script for Adversarial Patch Purification System
Demonstrates how to use the purification system with different attack types
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from purification import AdversarialPatchPurifier

def create_sample_adversarial_image(size=(256, 256, 3), attack_type='pgd'):
    """Create a sample image with simulated adversarial patches for testing."""
    # Create a base image
    base_image = np.random.randint(50, 200, size, dtype=np.uint8)
    
    # Add some structure to make it more realistic
    for i in range(0, size[0], 20):
        for j in range(0, size[1], 20):
            base_image[i:i+10, j:j+10] = np.random.randint(100, 150, (10, 10, 3))
    
    # Add simulated adversarial patches based on attack type
    if attack_type == 'pgd':
        # PGD: High gradient regions
        patch1 = np.random.randint(0, 50, (30, 30, 3))
        patch2 = np.random.randint(200, 255, (25, 25, 3))
        base_image[50:80, 50:80] = patch1
        base_image[150:175, 150:175] = patch2
        
    elif attack_type == 'fgsm':
        # FGSM: Noise patterns
        noise = np.random.normal(0, 30, (40, 40, 3))
        base_image[100:140, 100:140] = np.clip(
            base_image[100:140, 100:140].astype(float) + noise, 0, 255
        ).astype(np.uint8)
        
    elif attack_type == 'cw':
        # CW: L2 optimized patches
        patch = np.random.randint(0, 255, (35, 35, 3))
        # Add some structure to make it look more like a real patch
        patch[10:25, 10:25] = [255, 0, 0]  # Red square
        patch[5:15, 5:15] = [0, 255, 0]    # Green square
        base_image[80:115, 80:115] = patch
        
    elif attack_type == 'bpda':
        # BPDA: Backpropagation artifacts
        for i in range(60, 90):
            for j in range(60, 90):
                if (i + j) % 3 == 0:
                    base_image[i, j] = [255, 255, 255]
                elif (i + j) % 3 == 1:
                    base_image[i, j] = [0, 0, 0]
                    
    elif attack_type == 'eot':
        # EOT: Expectation-based patches
        patch = np.random.randint(100, 200, (30, 30, 3))
        # Add some variance to simulate expectation
        variance = np.random.normal(0, 20, (30, 30, 3))
        patch = np.clip(patch.astype(float) + variance, 0, 255).astype(np.uint8)
        base_image[120:150, 120:150] = patch
    
    return base_image

def demonstrate_purification():
    """Demonstrate the purification system with different attack types."""
    purifier = AdversarialPatchPurifier()
    
    print("Adversarial Patch Purification Demonstration")
    print("=" * 60)
    
    # Test with different attack types
    attack_types = ['pgd', 'fgsm', 'cw', 'bpda', 'eot']
    
    for attack_type in attack_types:
        print(f"\n{'='*20} {attack_type.upper()} Attack {'='*20}")
        
        # Create sample adversarial image
        adversarial_image = create_sample_adversarial_image(attack_type=attack_type)
        
        # Save the adversarial image
        purifier.save_image(adversarial_image, f"adversarial_{attack_type}.jpg")
        print(f"Created adversarial image: adversarial_{attack_type}.jpg")
        
        # Purify the image
        try:
            purified_image, patch_mask = purifier.purify_image(
                adversarial_image, 
                attack_type=attack_type
            )
            
            # Save the purified image
            purifier.save_image(purified_image, f"purified_{attack_type}.jpg")
            print(f"Purified image saved: purified_{attack_type}.jpg")
            
            # Calculate statistics
            patch_pixels = np.sum(patch_mask)
            total_pixels = patch_mask.size
            patch_percentage = (patch_pixels / total_pixels) * 100
            
            print(f"Detected patches: {patch_pixels} pixels ({patch_percentage:.2f}%)")
            
            # Calculate improvement metrics
            mse_original = np.mean((adversarial_image.astype(float) - 
                                  np.mean(adversarial_image))**2)
            mse_purified = np.mean((purified_image.astype(float) - 
                                  np.mean(purified_image))**2)
            
            print(f"Image variance reduction: {((mse_original - mse_purified) / mse_original * 100):.2f}%")
            
        except Exception as e:
            print(f"Error purifying {attack_type}: {str(e)}")

def demonstrate_auto_detection():
    """Demonstrate automatic attack detection."""
    purifier = AdversarialPatchPurifier()
    
    print(f"\n{'='*20} Automatic Detection {'='*20}")
    
    # Create a sample image
    sample_image = create_sample_adversarial_image(attack_type='pgd')
    
    # Try automatic detection
    try:
        purified_image, patch_mask = purifier.purify_image(
            sample_image, 
            attack_type='auto'
        )
        
        print("Automatic detection completed successfully!")
        print(f"Detected patches: {np.sum(patch_mask)} pixels")
        
        # Save results
        purifier.save_image(sample_image, "auto_detection_original.jpg")
        purifier.save_image(purified_image, "auto_detection_purified.jpg")
        
    except Exception as e:
        print(f"Automatic detection failed: {str(e)}")

def demonstrate_custom_parameters():
    """Demonstrate using custom parameters for purification."""
    purifier = AdversarialPatchPurifier()
    
    print(f"\n{'='*20} Custom Parameters {'='*20}")
    
    # Create sample image
    sample_image = create_sample_adversarial_image(attack_type='pgd')
    
    # Test with custom parameters
    custom_params = {
        'iterations': 5,  # More iterations for better purification
    }
    
    try:
        purified_image, patch_mask = purifier.purify_image(
            sample_image, 
            attack_type='pgd',
            **custom_params
        )
        
        print("Custom parameter purification completed!")
        print(f"Used {custom_params['iterations']} iterations")
        print(f"Detected patches: {np.sum(patch_mask)} pixels")
        
        # Save results
        purifier.save_image(sample_image, "custom_original.jpg")
        purifier.save_image(purified_image, "custom_purified.jpg")
        
    except Exception as e:
        print(f"Custom parameter purification failed: {str(e)}")

def create_comparison_plot():
    """Create a comparison plot showing original vs purified images."""
    purifier = AdversarialPatchPurifier()
    
    # Create sample images for different attacks
    attack_types = ['pgd', 'fgsm', 'cw']
    
    fig, axes = plt.subplots(2, len(attack_types), figsize=(15, 10))
    fig.suptitle('Adversarial Patch Purification Results', fontsize=16)
    
    for i, attack_type in enumerate(attack_types):
        # Create adversarial image
        adversarial_image = create_sample_adversarial_image(attack_type=attack_type)
        
        # Purify the image
        try:
            purified_image, patch_mask = purifier.purify_image(
                adversarial_image, 
                attack_type=attack_type
            )
            
            # Plot original
            axes[0, i].imshow(adversarial_image)
            axes[0, i].set_title(f'{attack_type.upper()} - Original')
            axes[0, i].axis('off')
            
            # Plot purified
            axes[1, i].imshow(purified_image)
            axes[1, i].set_title(f'{attack_type.upper()} - Purified')
            axes[1, i].axis('off')
            
        except Exception as e:
            print(f"Error creating comparison for {attack_type}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('purification_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved as 'purification_comparison.png'")

def main():
    """Main function to run all demonstrations."""
    print("Starting Adversarial Patch Purification Demonstrations...")
    
    # Run demonstrations
    demonstrate_purification()
    demonstrate_auto_detection()
    demonstrate_custom_parameters()
    
    # Create comparison plot
    try:
        create_comparison_plot()
    except Exception as e:
        print(f"Could not create comparison plot: {str(e)}")
    
    print("\n" + "="*60)
    print("All demonstrations completed!")
    print("Check the generated image files to see the results.")
    print("="*60)

if __name__ == "__main__":
    main()
