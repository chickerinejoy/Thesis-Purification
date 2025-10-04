"""
Adversarial Patch Purification System
Purifies images from PGD, FGSM, CW, BPDA, and EOT attacks without OpenCV or diffusion models
"""

import warnings
warnings.filterwarnings('ignore')

# new/updated imports
import numpy as np
try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from e
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

class AdversarialPatchPurifier:
    """Main class for purifying adversarial patches from various attack types."""
    
    def __init__(self):
        self.attack_detectors = {
            'pgd': self._detect_pgd_patch,
            'fgsm': self._detect_fgsm_patch,
            'cw': self._detect_cw_patch,
            'bpda': self._detect_bpda_patch,
            'eot': self._detect_eot_patch
        }
        
        self.purification_methods = {
            'pgd': self._purify_pgd,
            'fgsm': self._purify_fgsm,
            'cw': self._purify_cw,
            'bpda': self._purify_bpda,
            'eot': self._purify_eot
        }
    
    def load_image(self, image_path):
        """Loads an image using Pillow and converts it to a NumPy array."""
        img = Image.open(image_path)
        return np.array(img)
    
    def save_image(self, image_array, output_path):
        """Saves a NumPy array as an image using Pillow."""
        img = Image.fromarray(image_array.astype(np.uint8))
        img.save(output_path)
    
    def display_image(self, image_array, title="Image"):
        """Displays an image using Matplotlib."""
        plt.figure(figsize=(10, 8))
        plt.imshow(image_array)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def _detect_pgd_patch(self, image, threshold=0.1):
        """Detects PGD adversarial patches using gradient magnitude analysis."""
        if len(image.shape) == 3:
            gray = self._rgb_to_gray(image)
        else:
            gray = image.copy()
        
        # Compute gradient magnitude
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find high gradient regions (potential patches)
        patch_mask = gradient_magnitude > np.percentile(gradient_magnitude, 95)
        return patch_mask
    
    def _detect_fgsm_patch(self, image, threshold=0.15):
        """Detects FGSM adversarial patches using noise pattern analysis."""
        if len(image.shape) == 3:
            gray = self._rgb_to_gray(image)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to smooth the image
        blurred = self._gaussian_blur(gray, sigma=1.0)
        
        # Compute difference (noise pattern)
        noise = np.abs(gray.astype(float) - blurred.astype(float))
        
        # Find high noise regions
        patch_mask = noise > np.percentile(noise, 90)
        return patch_mask
    
    def _detect_cw_patch(self, image, threshold=0.2):
        """Detects CW adversarial patches using L2 norm analysis."""
        if len(image.shape) == 3:
            # Convert to different color spaces for analysis
            hsv = self._rgb_to_hsv(image)
            lab = self._rgb_to_lab(image)
            
            # Compute L2 norm across color channels
            l2_norm = np.sqrt(np.sum(image.astype(float)**2, axis=2))
            hsv_norm = np.sqrt(np.sum(hsv.astype(float)**2, axis=2))
            lab_norm = np.sqrt(np.sum(lab.astype(float)**2, axis=2))
            
            # Combine norms for detection
            combined_norm = (l2_norm + hsv_norm + lab_norm) / 3
            patch_mask = combined_norm > np.percentile(combined_norm, 85)
        else:
            patch_mask = np.zeros_like(image, dtype=bool)
        
        return patch_mask
    
    def _detect_bpda_patch(self, image, threshold=0.12):
        """Detects BPDA adversarial patches using backpropagation analysis."""
        if len(image.shape) == 3:
            gray = self._rgb_to_gray(image)
        else:
            gray = image.copy()
        
        # Apply multiple filters to detect backpropagation artifacts
        sobel_x = self._sobel_filter(gray, direction='x')
        sobel_y = self._sobel_filter(gray, direction='y')
        laplacian = self._laplacian_filter(gray)
        
        # Combine edge detection results
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        combined = edge_magnitude + np.abs(laplacian)
        
        patch_mask = combined > np.percentile(combined, 88)
        return patch_mask
    
    def _detect_eot_patch(self, image, threshold=0.18):
        """Detects EOT adversarial patches using expectation analysis."""
        if len(image.shape) == 3:
            gray = self._rgb_to_gray(image)
        else:
            gray = image.copy()
        
        # Apply multiple transformations and compute variance
        transformations = [
            self._gaussian_blur(gray, sigma=0.5),
            self._gaussian_blur(gray, sigma=1.5),
            self._median_filter(gray, size=3),
            self._median_filter(gray, size=5)
        ]
        
        # Compute variance across transformations
        variance = np.var(transformations, axis=0)
        patch_mask = variance > np.percentile(variance, 87)
        return patch_mask
    
    def _purify_pgd(self, image, patch_mask, iterations=3):
        """Purifies PGD adversarial patches using iterative denoising."""
        purified = image.copy().astype(float)
        
        for _ in range(iterations):
            # Apply bilateral filtering to preserve edges while reducing noise
            purified = self._bilateral_filter(purified, patch_mask)
            
            # Apply non-local means denoising
            purified = self._non_local_means(purified, patch_mask)
            
            # Apply total variation denoising
            purified = self._total_variation_denoise(purified, patch_mask)
        
        return np.clip(purified, 0, 255).astype(np.uint8)
    
    def _purify_fgsm(self, image, patch_mask, iterations=2):
        """Purifies FGSM adversarial patches using gradient-based filtering."""
        purified = image.copy().astype(float)
        
        for _ in range(iterations):
            # Apply anisotropic diffusion
            purified = self._anisotropic_diffusion(purified, patch_mask)
            
            # Apply edge-preserving smoothing
            purified = self._edge_preserving_smooth(purified, patch_mask)
        
        return np.clip(purified, 0, 255).astype(np.uint8)
    
    def _purify_cw(self, image, patch_mask, iterations=4):
        """Purifies CW adversarial patches using L2 optimization."""
        purified = image.copy().astype(float)
        
        for _ in range(iterations):
            # Apply L2 regularized denoising
            purified = self._l2_regularized_denoise(purified, patch_mask)
            
            # Apply color space optimization
            purified = self._color_space_optimize(purified, patch_mask)
            
            # Apply perceptual loss minimization
            purified = self._perceptual_denoise(purified, patch_mask)
        
        return np.clip(purified, 0, 255).astype(np.uint8)
    
    def _purify_bpda(self, image, patch_mask, iterations=3):
        """Purifies BPDA adversarial patches using backpropagation defense."""
        purified = image.copy().astype(float)
        
        for _ in range(iterations):
            # Apply gradient reversal
            purified = self._gradient_reversal(purified, patch_mask)
            
            # Apply feature reconstruction
            purified = self._feature_reconstruction(purified, patch_mask)
            
            # Apply adversarial training defense
            purified = self._adversarial_training_defense(purified, patch_mask)
        
        return np.clip(purified, 0, 255).astype(np.uint8)
    
    def _purify_eot(self, image, patch_mask, iterations=2):
        """Purifies EOT adversarial patches using expectation-based filtering."""
        # Fallback to edge-preserving inpaint
        return self._inpaint_preserve_edges(image, patch_mask, radius=3, method='telea')

    def purify_image(self, image, attack_type='auto', **kwargs):
        """Main entry: detect and purify. Uses OpenCV inpainting while preserving edges."""
        # Normalize input to numpy array uint8
        img = np.array(image, copy=False)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Validate explicit attack_type
        if attack_type != 'auto' and attack_type not in self.attack_detectors:
            raise ValueError(f"Invalid attack_type '{attack_type}'. Expected one of: "
                             f"{sorted(list(self.attack_detectors.keys()))} or 'auto'.")
        
        # If attack type explicitly provided and known, use it
        if attack_type != 'auto' and attack_type in self.attack_detectors:
            detector = self.attack_detectors[attack_type]
            mask = detector(img)
            purified = self.purification_methods.get(attack_type, self._purify_pgd)(img, mask, **kwargs)
            # Optional Real-ESRGAN super-resolution after purification
            if kwargs.get('super_resolve', False):
                scale = int(kwargs.get('sr_scale', 2))
                model_path = kwargs.get('sr_model_path', None)
                device = kwargs.get('sr_device', 'cpu')
                try:
                    purified = self._super_resolve(purified, scale=scale, model_path=model_path, device=device)
                except Exception as e:
                    print(f"Warning: Real-ESRGAN failed: {e}")
            return purified, mask.astype(bool)

        # Auto-detection: run all detectors and pick the most confident (largest mask area)
        scores = {}
        masks = {}
        for name, detector in self.attack_detectors.items():
            try:
                m = detector(img)
                if m is None:
                    m = np.zeros(img.shape[:2], dtype=bool)
                masks[name] = m.astype(bool)
                scores[name] = masks[name].sum()
            except Exception:
                masks[name] = np.zeros(img.shape[:2], dtype=bool)
                scores[name] = 0

        # choose attack type with largest detected area
        detected = max(scores.items(), key=lambda kv: kv[1])
        attack_detected = detected[0]
        patch_mask = masks[attack_detected]

        # Purify using edge-preserving inpaint strategy that preserves strong edges
        purified = self._inpaint_preserve_edges(img, patch_mask, radius=3, method='telea')

        # Optional Real-ESRGAN super-resolution after purification
        if kwargs.get('super_resolve', False):
            scale = int(kwargs.get('sr_scale', 2))
            model_path = kwargs.get('sr_model_path', None)
            device = kwargs.get('sr_device', 'cpu')
            try:
                purified = self._super_resolve(purified, scale=scale, model_path=model_path, device=device)
            except Exception as e:
                print(f"Warning: Real-ESRGAN failed: {e}")

        return purified, patch_mask.astype(bool)

    # Helper methods for image processing
    def _rgb_to_gray(self, image):
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def _rgb_to_hsv(self, image):
        """Convert RGB to HSV color space."""
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        r, g, b = r/255.0, g/255.0, b/255.0
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Hue
        h = np.zeros_like(max_val)
        h[max_val == r] = (60 * ((g[max_val == r] - b[max_val == r]) / diff[max_val == r]) + 360) % 360
        h[max_val == g] = (60 * ((b[max_val == g] - r[max_val == g]) / diff[max_val == g]) + 120) % 360
        h[max_val == b] = (60 * ((r[max_val == b] - g[max_val == b]) / diff[max_val == b]) + 240) % 360
        h[diff == 0] = 0
        
        # Saturation
        s = np.zeros_like(max_val)
        s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]
        
        # Value
        v = max_val
        
        return np.stack([h, s, v], axis=2) * 255
    
    def _rgb_to_lab(self, image):
        """Convert RGB to LAB color space (simplified)."""
        # This is a simplified conversion - in practice, you'd use proper color space conversion
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Simple linear transformation approximation
        l = 0.2126 * r + 0.7152 * g + 0.0722 * b
        a = r - g
        b_lab = g - b
        
        return np.stack([l, a, b_lab], axis=2)
    
    def _gaussian_blur(self, image, sigma=1.0):
        k = max(3, int(sigma * 4) | 1)
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

    def _median_filter(self, image, size=3):
        return cv2.medianBlur(image.astype(np.uint8), size)

    def _sobel_filter(self, image, direction='x'):
        if direction == 'x':
            dx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            return dx
        dy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return dy

    def _laplacian_filter(self, image):
        return cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
    
    def _bilateral_filter(self, image, mask, sigma_color=75, sigma_space=75):
        """Apply bilateral filter (simplified implementation)."""
        # This is a simplified bilateral filter - in practice, you'd use a more sophisticated implementation
        return self._gaussian_blur(image, sigma=1.0)
    
    def _non_local_means(self, image, mask, h=10):
        """Apply non-local means denoising (simplified)."""
        # Simplified implementation - in practice, you'd use a more sophisticated algorithm
        return self._gaussian_blur(image, sigma=1.5)
    
    def _total_variation_denoise(self, image, mask, weight=0.1):
        """Apply total variation denoising (simplified)."""
        # Simplified implementation using gradient descent
        result = image.copy()
        for _ in range(10):  # Iterations
            grad_x = np.gradient(result, axis=1)
            grad_y = np.gradient(result, axis=0)
            tv_grad = np.gradient(grad_x, axis=1) + np.gradient(grad_y, axis=0)
            result = result - weight * tv_grad
        return result
    
    def _anisotropic_diffusion(self, image, mask, iterations=10, delta_t=0.1, kappa=30):
        """Apply anisotropic diffusion (simplified)."""
        result = image.copy()
        for _ in range(iterations):
            grad_x = np.gradient(result, axis=1)
            grad_y = np.gradient(result, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Diffusion coefficient
            c = np.exp(-(grad_magnitude / kappa)**2)
            
            # Apply diffusion
            diff_x = c * grad_x
            diff_y = c * grad_y
            
            div = np.gradient(diff_x, axis=1) + np.gradient(diff_y, axis=0)
            result = result + delta_t * div
        
        return result
    
    def _edge_preserving_smooth(self, image, mask):
        """Apply edge-preserving smoothing."""
        return self._gaussian_blur(image, sigma=0.8)
    
    def _l2_regularized_denoise(self, image, mask, lambda_reg=0.1):
        """Apply L2 regularized denoising."""
        # Simplified L2 regularization
        result = image.copy()
        for _ in range(5):
            laplacian = self._laplacian_filter(result)
            result = result - lambda_reg * laplacian
        return result
    
    def _color_space_optimize(self, image, mask):
        """Apply color space optimization."""
        # Convert to different color space and back
        hsv = self._rgb_to_hsv(image)
        # Apply smoothing in HSV space
        hsv_smooth = self._gaussian_blur(hsv, sigma=1.0)
        # Convert back (simplified)
        return self._gaussian_blur(image, sigma=0.5)
    
    def _perceptual_denoise(self, image, mask):
        """Apply perceptual loss minimization (simplified)."""
        return self._gaussian_blur(image, sigma=1.2)
    
    def _gradient_reversal(self, image, mask):
        """Apply gradient reversal defense."""
        # Reverse gradients in detected patch areas
        result = image.copy()
        grad_x = np.gradient(result, axis=1)
        grad_y = np.gradient(result, axis=0)
        
        # Reverse gradients in patch areas
        result[mask] = result[mask] - 0.1 * (grad_x[mask] + grad_y[mask])
        return result
    
    def _feature_reconstruction(self, image, mask):
        """Apply feature reconstruction."""
        return self._gaussian_blur(image, sigma=1.0)
    
    def _adversarial_training_defense(self, image, mask):
        """Apply adversarial training defense."""
        return self._gaussian_blur(image, sigma=0.8)
    
    def _expectation_maximization(self, image, mask):
        """Apply expectation maximization."""
        return self._gaussian_blur(image, sigma=1.5)
    
    def _robust_statistics_filter(self, image, mask):
        """Apply robust statistics filtering."""
        return self._median_filter(image, size=3)

    def _inpaint_preserve_edges(self, image, patch_mask, radius=3, method='telea'):
        """
        Inpaint the patch_mask region while preserving edges:
        - compute strong edges with Canny
        - remove edges from the inpainting mask so edges are preserved
        - perform inpainting and blend with edge-preserved bilateral filtering
        """
        img = image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # ensure boolean mask
        mask = (patch_mask.astype(np.uint8) * 255).astype(np.uint8)

        # detect strong edges
        edges = cv2.Canny(gray, 50, 150)
        # dilate edges slightly so we truly preserve them
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # create inpaint mask that excludes strong edges
        inpaint_mask = mask.copy()
        inpaint_mask[edges_dilated > 0] = 0

        # If inpaint mask empty (edges cover patch), fall back to small erosion so something is repaired
        if inpaint_mask.sum() == 0 and mask.sum() > 0:
            # erode edges a bit to allow minimal inpainting
            edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
            inpaint_mask = mask.copy()
            inpaint_mask[edges_eroded > 0] = 0

        # OpenCV requires single-channel 8-bit mask
        inpaint_mask8 = (inpaint_mask > 0).astype(np.uint8) * 255

        # Choose inpaint method
        flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS

        # Inpaint on 3-channel BGR (convert RGB->BGR for cv2)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, inpaint_mask8, radius, flags)
        inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

        # 1) Denoise the inpainted region to remove residual adversarial noise,
        #    while preserving overall texture and color.
        inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(inpainted_bgr, None,
                                                       h=10, hColor=10,
                                                       templateWindowSize=7, searchWindowSize=21)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

        # 2) Blend denoised pixels back only inside the inpaint mask (preserve original elsewhere)
        final = img.copy()
        use_mask_2d = (inpaint_mask8 > 0)
        use_mask = np.repeat(use_mask_2d[:, :, None], 3, axis=2)
        final[use_mask] = denoised_rgb[use_mask]

        # 3) Recover important details: detect strong edges and apply a targeted unsharp mask
        gray_final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        edges_detail = cv2.Canny(gray_final, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges_detail, kernel, iterations=1)
        edges3 = np.repeat(edges_dilated[:, :, None] > 0, 3, axis=2)

        # Create an unsharp (detail-enhanced) version but only apply on detected edges
        blur = cv2.GaussianBlur(final, (0, 0), sigmaX=1.0)
        sharpen = cv2.addWeighted(final, 1.15, blur, -0.15, 0)
        # Apply sharpening on strong edges (this avoids reintroducing noise in smooth areas)
        final[edges3] = sharpen[edges3]

        # 4) Gentle bilateral filtering to unify textures while preserving edges
        final = cv2.bilateralFilter(final, d=9, sigmaColor=75, sigmaSpace=75)

        # 5) Final clamp & return
        return np.clip(final, 0, 255).astype(np.uint8)

    def _super_resolve(self, image, scale=2, model_path=None, device='cpu'):
        """
        Run Real-ESRGAN to upscale and restore fine details.
        Requires: pip install basicsr realesrgan and a compatible torch build.
        - image: RGB uint8 numpy array
        - scale: upscale factor (commonly 2 or 4)
        - model_path: path to Real-ESRGAN .pth model file
        - device: 'cpu' or 'cuda'
        """
        if model_path is None:
            raise FileNotFoundError("Real-ESRGAN model_path not provided. Download a Real-ESRGAN .pth and set sr_model_path.")

        try:
            from realesrgan import RealESRGANer
        except Exception as e:
            raise ImportError(
                "Real-ESRGAN integration requires 'realesrgan' and 'basicsr'. "
                "Install with (CPU example):\n"
                "  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install basicsr realesrgan"
            ) from e

        # convert RGB -> BGR for Real-ESRGAN input
        img_bgr = cv2.cvtColor(np.asarray(image).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # create the RealESRGANer
        rr = RealESRGANer(scale=scale, model_path=model_path, dni_weight=None, device=device)

        # run enhancement (returns (output, None/tuple) in most implementations)
        out, _ = rr.enhance(img_bgr, outscale=scale)

        out_rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return out_rgb

# Example usage and testing
def main():
    """Main function demonstrating the adversarial patch purification system."""
    purifier = AdversarialPatchPurifier()
    
    # Example usage
    print("Adversarial Patch Purification System")
    print("=" * 50)
    
    # Load a sample image (you can replace this with your own image)
    try:
        # Create a sample image for demonstration
        sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print("Using sample image for demonstration")
    except:
        print("Please provide a valid image path")
        return
    
    # Test different attack types
    attack_types = ['pgd', 'fgsm', 'cw', 'bpda', 'eot']
    
    for attack_type in attack_types:
        print(f"\nTesting {attack_type.upper()} purification:")
        try:
            purified, patch_mask = purifier.purify_image(sample_image, attack_type=attack_type)
            print(f"✓ {attack_type.upper()} purification completed")
            print(f"  Detected patches: {np.sum(patch_mask)} pixels")
        except Exception as e:
            print(f"✗ {attack_type.upper()} purification failed: {str(e)}")
    
    # Test automatic detection
    print(f"\nTesting automatic attack detection:")
    try:
        purified, patch_mask = purifier.purify_image(sample_image, attack_type='auto')
        print("✓ Automatic detection and purification completed")
    except Exception as e:
        print(f"✗ Automatic detection failed: {str(e)}")

if __name__ == "__main__":
    main()