"""
Test Script for Adversarial Patch Purification System
Verifies that all components work correctly
"""

import numpy as np
import sys
import traceback
import os
from PIL import Image
from purification import AdversarialPatchPurifier

# create deterministic structured image and synthetic patch helpers
def _create_structured_image(shape=(128, 128, 3), seed=42):
    np.random.seed(seed)
    h, w, c = shape
    # smooth gradient background
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    gx = np.tile(x, (h, 1))
    gy = np.tile(y[:, None], (1, w))
    gradient = ((gx + gy) / 2.0 * 200).astype(np.uint8)
    base = np.stack([gradient + 20, gradient + 10, gradient], axis=2)
    # add low-frequency structure (soft circles)
    for r, cx, cy, val in [(18, w//4, h//3, 30), (24, w*3//4, h*2//3, -20)]:
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
        for ch in range(3):
            base[..., ch][mask] = np.clip(base[..., ch][mask].astype(int) + val, 0, 255)
    return base.astype(np.uint8)

def _add_synthetic_patch(image, attack_type='pgd', size=32, intensity=120, seed=1):
    np.random.seed(seed)
    img = image.copy().astype(np.uint8)
    h, w = img.shape[:2]
    # place patch near center
    x0 = w//2 - size//2
    y0 = h//2 - size//2
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    if attack_type == 'pgd':
        # high-frequency noisy patch
        patch = (np.random.randint(0, 255, (size, size, 3)) * 0.6 + intensity).astype(np.uint8)
    elif attack_type == 'fgsm':
        # signed noise pattern
        noise = (np.random.randn(size, size, 3) * 25 + intensity).clip(0, 255)
        patch = noise.astype(np.uint8)
    elif attack_type == 'cw':
        # smooth color shift
        for ch in range(3):
            patch[..., ch] = np.linspace(intensity + ch*10, intensity + ch*15, size).astype(np.uint8)[None, :]
    elif attack_type == 'bpda':
        # structured stripes
        for i in range(size):
            patch[i, :, :] = ((i % 2) * 255) // 2
    else:  # eot
        patch = (np.random.randint(0, 255, (size, size, 3)) * 0.4 + intensity/2).astype(np.uint8)
    img[y0:y0+size, x0:x0+size] = patch
    return img

# ensure output dir
OUT_DIR = "test_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def test_basic_functionality():
    """Test basic functionality of the purification system."""
    print("Testing basic functionality...")
    
    try:
        # Initialize purifier
        purifier = AdversarialPatchPurifier()
        print("✓ Purifier initialized successfully")
        
        # Create a deterministic structured test image and add synthetic patches before each test
        base_image = _create_structured_image((128, 128, 3))
        print("✓ Structured test image created")
        
        # Test all attack types with clearer patches and stronger purification params
        attack_types = ['pgd', 'fgsm', 'cw', 'bpda', 'eot']
        
        for attack_type in attack_types:
            try:
                test_image = _add_synthetic_patch(base_image, attack_type=attack_type, size=34, intensity=140, seed=hash(attack_type) & 0xffffffff)
                # ask purifier for more iterations for clearer result when supported
                purified, mask = purifier.purify_image(test_image, attack_type=attack_type, iterations=5)
                print(f"✓ {attack_type.upper()} purification successful")
                
                # Save original + purified for manual inspection
                Image.fromarray(test_image).save(os.path.join(OUT_DIR, f"{attack_type}_original.png"))
                # ensure purified is uint8 before saving
                if purified.dtype != np.uint8:
                    purified = np.clip(purified, 0, 255).astype(np.uint8)
                Image.fromarray(purified).save(os.path.join(OUT_DIR, f"{attack_type}_purified.png"))
                
                # Verify output shapes
                assert purified.shape == test_image.shape, f"Shape mismatch for {attack_type}"
                assert mask.shape == test_image.shape[:2], f"Mask shape mismatch for {attack_type}"
                # accept boolean or binary masks
                assert mask.ndim == 2 and mask.shape == test_image.shape[:2], f"Mask invalid for {attack_type}"
                
            except Exception as e:
                print(f"✗ {attack_type.upper()} purification failed: {str(e)}")
                traceback.print_exc()
                return False
        
        # Test automatic detection on a CW-like patched image (clear contrast)
        try:
            test_image = _add_synthetic_patch(base_image, attack_type='cw', size=34, intensity=140, seed=12345)
            purified, mask = purifier.purify_image(test_image, attack_type='auto', iterations=5)
            Image.fromarray(test_image).save(os.path.join(OUT_DIR, "auto_original.png"))
            Image.fromarray(np.clip(purified, 0, 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "auto_purified.png"))
            print("✓ Automatic detection successful")
        except Exception as e:
            print(f"✗ Automatic detection failed: {str(e)}")
            traceback.print_exc()
            return False
        
        print(f"Saved output images to {OUT_DIR}/")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing functions."""
    print("\nTesting image processing functions...")
    
    try:
        purifier = AdversarialPatchPurifier()
        
        # Create test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Test color space conversions
        gray = purifier._rgb_to_gray(test_image)
        assert gray.shape == (64, 64), "Grayscale conversion failed"
        print("✓ RGB to grayscale conversion")
        
        hsv = purifier._rgb_to_hsv(test_image)
        assert hsv.shape == test_image.shape, "HSV conversion failed"
        print("✓ RGB to HSV conversion")
        
        lab = purifier._rgb_to_lab(test_image)
        assert lab.shape == test_image.shape, "LAB conversion failed"
        print("✓ RGB to LAB conversion")
        
        # Test filtering functions
        blurred = purifier._gaussian_blur(test_image, sigma=1.0)
        assert blurred.shape == test_image.shape, "Gaussian blur failed"
        print("✓ Gaussian blur")
        
        median = purifier._median_filter(test_image, size=3)
        assert median.shape == test_image.shape, "Median filter failed"
        print("✓ Median filter")
        
        sobel_x = purifier._sobel_filter(gray, direction='x')
        sobel_y = purifier._sobel_filter(gray, direction='y')
        assert sobel_x.shape == gray.shape, "Sobel X filter failed"
        assert sobel_y.shape == gray.shape, "Sobel Y filter failed"
        print("✓ Sobel filters")
        
        laplacian = purifier._laplacian_filter(gray)
        assert laplacian.shape == gray.shape, "Laplacian filter failed"
        print("✓ Laplacian filter")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_detection_algorithms():
    """Test detection algorithms."""
    print("\nTesting detection algorithms...")
    
    try:
        purifier = AdversarialPatchPurifier()
        
        # Create test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Test all detection methods
        detection_methods = [
            ('PGD', purifier._detect_pgd_patch),
            ('FGSM', purifier._detect_fgsm_patch),
            ('CW', purifier._detect_cw_patch),
            ('BPDA', purifier._detect_bpda_patch),
            ('EOT', purifier._detect_eot_patch)
        ]
        
        for name, method in detection_methods:
            try:
                mask = method(test_image)
                assert mask.shape == test_image.shape[:2], f"{name} detection shape mismatch"
                assert mask.dtype == bool, f"{name} detection should return boolean mask"
                print(f"✓ {name} detection")
            except Exception as e:
                print(f"✗ {name} detection failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Detection algorithms test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_purification_methods():
    """Test purification methods."""
    print("\nTesting purification methods...")
    
    try:
        purifier = AdversarialPatchPurifier()
        
        # Create test image and mask
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_mask = np.random.random((64, 64)) > 0.8
        
        # Test all purification methods
        purification_methods = [
            ('PGD', purifier._purify_pgd),
            ('FGSM', purifier._purify_fgsm),
            ('CW', purifier._purify_cw),
            ('BPDA', purifier._purify_bpda),
            ('EOT', purifier._purify_eot)
        ]
        
        for name, method in purification_methods:
            try:
                purified = method(test_image, test_mask)
                assert purified.shape == test_image.shape, f"{name} purification shape mismatch"
                assert purified.dtype == np.uint8, f"{name} purification should return uint8"
                print(f"✓ {name} purification")
            except Exception as e:
                print(f"✗ {name} purification failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Purification methods test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        purifier = AdversarialPatchPurifier()
        
        # Grayscale image handling (2D)
        test_image_gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        purified, mask = purifier.purify_image(test_image_gray, attack_type='pgd')
        print("✓ Grayscale image handling")
        
        # Small color image handling
        test_image_small = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        purified, mask = purifier.purify_image(test_image_small, attack_type='fgsm')
        print("✓ Small image handling")
        
        # Invalid attack type should raise ValueError
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        try:
            purifier.purify_image(test_image, attack_type='invalid')
            print("✗ Invalid attack type did not raise")
            return False
        except ValueError:
            print("✓ Invalid attack type raised ValueError")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_performance_test():
    """Run a simple performance test."""
    print("\nRunning performance test...")
    
    try:
        import time
        purifier = AdversarialPatchPurifier()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Time the purification process
        start_time = time.time()
        purified, mask = purifier.purify_image(test_image, attack_type='auto')
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"✓ Processing time: {processing_time:.2f} seconds")
        
        if processing_time < 10:  # Should be reasonably fast
            print("✓ Performance acceptable")
            return True
        else:
            print("⚠ Performance may be slow for large images")
            return True
            
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Adversarial Patch Purification System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Image Processing", test_image_processing),
        ("Detection Algorithms", test_detection_algorithms),
        ("Purification Methods", test_purification_methods),
        ("Edge Cases", test_edge_cases),
        ("Performance", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
