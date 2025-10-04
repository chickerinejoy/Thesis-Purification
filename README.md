# Adversarial Patch Purification System

A comprehensive Python system for purifying adversarial patches from various attack types (PGD, FGSM, CW, BPDA, EOT). The project can optionally run Real-ESRGAN for super-resolution after purification.

## Features

- **Multi-Attack Support**: Handles PGD, FGSM, CW, BPDA, and EOT adversarial attacks
- **Automatic Detection**: Automatically detects the type of adversarial attack
- **Customizable Purification**: Adjustable parameters for different purification strategies
- **No External Dependencies**: Uses only NumPy, PIL, Matplotlib, and SciPy
- **Comprehensive Detection**: Multiple detection algorithms for different attack patterns
- **Advanced Purification**: Specialized purification methods for each attack type

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The system is ready to use!

## Quick Start

```python
from purification import AdversarialPatchPurifier

# Initialize the purifier
purifier = AdversarialPatchPurifier()

# Load an image
image = purifier.load_image("your_image.jpg")

# Purify with automatic attack detection
purified_image, patch_mask = purifier.purify_image(image, attack_type='auto')

# Save the purified image
purifier.save_image(purified_image, "purified_image.jpg")
```

## Usage Examples

### Basic Usage

```python
from purification import AdversarialPatchPurifier

purifier = AdversarialPatchPurifier()

# Load image
image = purifier.load_image("adversarial_image.jpg")

# Purify with specific attack type
purified, mask = purifier.purify_image(image, attack_type='pgd')

# Display results
purifier.display_image(image, "Original")
purifier.display_image(purified, "Purified")
```

### Automatic Attack Detection

```python
# Let the system automatically detect the attack type
purified, mask = purifier.purify_image(image, attack_type='auto')
print(f"Detected patches: {np.sum(mask)} pixels")
```

### Custom Parameters

```python
# Use custom parameters for purification
custom_params = {
    'iterations': 5,  # More iterations for better results
}

purified, mask = purifier.purify_image(
    image, 
    attack_type='pgd', 
    **custom_params
)
```

## Supported Attack Types

### 1. PGD (Projected Gradient Descent)
- **Detection**: Gradient magnitude analysis
- **Purification**: Iterative denoising with bilateral filtering, non-local means, and total variation denoising

### 2. FGSM (Fast Gradient Sign Method)
- **Detection**: Noise pattern analysis
- **Purification**: Anisotropic diffusion and edge-preserving smoothing

### 3. CW (Carlini & Wagner)
- **Detection**: L2 norm analysis across color spaces
- **Purification**: L2 regularized denoising, color space optimization, and perceptual loss minimization

### 4. BPDA (Backward Pass Differentiable Approximation)
- **Detection**: Backpropagation artifact analysis
- **Purification**: Gradient reversal, feature reconstruction, and adversarial training defense

### 5. EOT (Expectation Over Transformation)
- **Detection**: Expectation analysis across transformations
- **Purification**: Expectation maximization and robust statistics filtering

## API Reference

### AdversarialPatchPurifier Class

#### Methods

- `load_image(image_path)`: Load an image from file
- `save_image(image_array, output_path)`: Save an image to file
- `display_image(image_array, title)`: Display an image using matplotlib
- `purify_image(image, attack_type='auto', **kwargs)`: Main purification function

#### Parameters

- `image`: Input image as numpy array
- `attack_type`: Type of attack ('auto', 'pgd', 'fgsm', 'cw', 'bpda', 'eot')
- `**kwargs`: Additional parameters for purification methods

## Example Scripts

Run the example script to see the system in action:

```bash
python example_usage.py
```

This will:
- Create sample adversarial images for each attack type
- Demonstrate purification for each attack
- Show automatic detection capabilities
- Generate comparison plots
- Save results as image files

## Technical Details

### Detection Algorithms

Each attack type uses specialized detection algorithms:

- **PGD**: Gradient magnitude analysis to detect high-gradient regions
- **FGSM**: Noise pattern analysis using Gaussian blur differences
- **CW**: L2 norm analysis across RGB, HSV, and LAB color spaces
- **BPDA**: Combined Sobel and Laplacian edge detection
- **EOT**: Variance analysis across multiple image transformations

### Purification Methods

Purification methods are tailored to each attack type:

- **PGD**: Iterative denoising with bilateral filtering and total variation
- **FGSM**: Anisotropic diffusion and edge-preserving smoothing
- **CW**: L2 regularized denoising and color space optimization
- **BPDA**: Gradient reversal and feature reconstruction
- **EOT**: Expectation maximization and robust statistics

## Performance Considerations

- The system is optimized for images up to 1024x1024 pixels
- Processing time scales with image size and number of iterations
- Memory usage is proportional to image size
- For large images, consider resizing before processing

## Limitations

- Simplified implementations of some advanced algorithms
- Performance may vary with different image types
- Detection accuracy depends on attack strength and image content
- Some methods are approximations of more complex algorithms

## Contributing

Feel free to contribute by:
- Improving detection algorithms
- Adding new attack types
- Optimizing purification methods
- Adding more sophisticated image processing techniques

## License

This project is open source and available under the MIT License.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{adversarial_patch_purification,
  title={Adversarial Patch Purification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/adversarial-patch-purification}
}
```

## Acknowledgments

- Based on research in adversarial machine learning
- Inspired by various defense mechanisms in computer vision
- Built using NumPy, PIL, Matplotlib, and SciPy

## Real-ESRGAN (optional) — super-resolution after purification

This project supports optional Real-ESRGAN upscaling to restore fine details after purification. Two integration options:

Option A — Install realesrgan (recommended)
- Install PyTorch first (choose CPU or CUDA build). Example (CPU):
  - python -m pip install --upgrade pip setuptools wheel
  - pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
- Install Real-ESRGAN Python packages:
  - pip install basicsr realesrgan
- Download a pretrained model that matches your desired scale and place it in your project, e.g.:
  - `c:\Users\paild\Desktop\Thesis Code\models\RealESRGAN_x4plus.pth`
  - Choose model: `RealESRGAN_x2plus.pth` (sr_scale=2) or `RealESRGAN_x4plus.pth` (sr_scale=4)
- Call the purifier with SR flags (see run_image.py example below).

Option B — Use the local Real-ESRGAN repository (no pip build)
- The repository copy is included at `Real-ESRGAN/`. You can:
  - Install editable: `pip install -e .\Real-ESRGAN` (may require build tooling), or
  - Use the included inference script without installing packages: the purifier falls back to calling `Real-ESRGAN\inference_realesrgan.py` if the `realesrgan` package is unavailable.
- Put model weights in `Real-ESRGAN/weights/` (or pass `--model_path` to the inference script).

Example: run_image.py (already included)
- Ensure you placed a model at `c:\Users\paild\Desktop\Thesis Code\models\RealESRGAN_x4plus.pth` (or use `Real-ESRGAN/weights/`).
- Example invocation from run_image.py:
```python
from purification import AdversarialPatchPurifier

purifier = AdversarialPatchPurifier()
img = purifier.load_image(r"c:\Users\paild\Desktop\Thesis Code\adversarial_image.jpeg")

purified, mask = purifier.purify_image(
    img,
    attack_type='auto',
    iterations=3,
    super_resolve=True,
    sr_scale=4,  # 2 for x2 model, 4 for x4 model
    sr_model_path=r"c:\Users\paild\Desktop\Thesis Code\models\RealESRGAN_x4plus.pth",
    sr_device='cpu'  # or 'cuda' if you have GPU-enabled torch
)

purifier.save_image(purified, r"c:\Users\paild\Desktop\Thesis Code\purified_knife_sr.png")
```

Troubleshooting
- `pip install basicsr realesrgan` may need build dependencies. If installation fails, use the local repo fallback described above.
- Model mismatch: set `sr_scale` to the model's scale (x2 vs x4).
- If using GPU, install a matching CUDA build of torch before installing `realesrgan`.
- If the Python API (`realesrgan`) is not available, the purifier will try the local `Real-ESRGAN/inference_realesrgan.py` script (requires the repo and model files).

Notes
- Super-resolution is optional — purification works without it.
- Placing models in `models/` (project root) or `Real-ESRGAN/weights/` is fine; pass `sr_model_path` when using the API or `--model_path` to the inference script.
