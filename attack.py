import argparse
import glob
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _add_synthetic_patch(image, attack_type='pgd', size=640, intensity=120, seed=1, opacity=0.35):
    """
    Add synthetic adversarial patch to an image (supports RGB, L, RGBA)
    'size' may be:
      - an integer >= 4: interpreted as pixels (patch generated at this size then resized to image)
      - a float between 0 and 1: interpreted as fraction of the image min dimension (converted to pixels)
    The patch is generated (default 640x640), resized to the image resolution and alpha-blended
    on top of the original image with the given opacity (0..1).
    Returns a new numpy uint8 image with same mode/channels as input.
    """
    np.random.seed(seed)
    img_arr = np.array(image).copy()

    # Handle grayscale (L) and RGBA
    has_alpha = False
    if img_arr.ndim == 2:
        # grayscale -> convert to 3-channel for processing then convert back
        img_rgb = np.stack([img_arr] * 3, axis=2)
        out_mode = 'L'
    elif img_arr.shape[2] == 4:
        has_alpha = True
        alpha = img_arr[..., 3].copy()
        img_rgb = img_arr[..., :3].copy()
        out_mode = 'RGBA'
    else:
        img_rgb = img_arr[..., :3].copy()
        out_mode = 'RGB'

    h, w = img_rgb.shape[:2]

    # interpret requested size: fractional or absolute pixels
    try:
        size_val = float(size)
    except Exception:
        size_val = 640.0

    if 0 < size_val <= 1.0:
        # fraction of the smaller image dimension -> compute target size then generate patch at that size
        requested = max(4, int(round(min(h, w) * size_val)))
    else:
        requested = max(4, int(round(size_val)))

    # generate base patch at requested size (default 640)
    base_size = requested

    # generate patch depending on attack_type
    if attack_type == 'pgd':
        base_patch = (np.random.randint(0, 255, (base_size, base_size, 3)) * 0.6 + intensity).astype(np.uint8)
    elif attack_type == 'fgsm':
        base_patch = (np.random.randn(base_size, base_size, 3) * 25 + intensity).clip(0, 255).astype(np.uint8)
    elif attack_type == 'cw':
        base_patch = np.zeros((base_size, base_size, 3), dtype=np.uint8)
        for ch in range(3):
            row = np.linspace(intensity + ch * 6, intensity + ch * 12, base_size).astype(np.uint8)
            base_patch[..., ch] = np.tile(row, (base_size, 1))
    elif attack_type == 'bpda':
        base_patch = np.zeros((base_size, base_size, 3), dtype=np.uint8)
        for i in range(base_size):
            val = 255 if (i % 2 == 0) else 0
            base_patch[i, :, :] = val
    elif attack_type == 'eot':
        base_patch = (np.random.randint(0, 255, (base_size, base_size, 3)) * 0.4 + intensity / 2).astype(np.uint8)
    else:
        base_patch = (np.random.randint(0, 255, (base_size, base_size, 3)) * 0.6 + intensity).astype(np.uint8)

    # use PIL for high-quality resizing
    from PIL import Image as PILImage
    patch_img = PILImage.fromarray(base_patch)
    patch_resized = np.asarray(patch_img.resize((w, h), resample=PILImage.BILINEAR)).astype(np.uint8)

    # blend patch_resized over img_rgb using opacity
    alpha = float(opacity)
    alpha = max(0.0, min(1.0, alpha))

    if out_mode == 'L':
        # convert patch to grayscale and blend
        patch_gray = (0.299 * patch_resized[..., 0] + 0.587 * patch_resized[..., 1] + 0.114 * patch_resized[..., 2]).astype(np.uint8)
        out = ((1.0 - alpha) * img_arr + alpha * patch_gray).clip(0, 255).astype(np.uint8)
    elif out_mode == 'RGBA':
        # if original has alpha, composite properly: blend RGB channels and keep original alpha
        blended_rgb = ((1.0 - alpha) * img_rgb + alpha * patch_resized).clip(0, 255).astype(np.uint8)
        out = np.dstack([blended_rgb, alpha * 255 * np.ones_like(alpha, dtype=np.uint8) if False else alpha_channel_placeholder()])  # placeholder replaced below
        # since we preserved original alpha, use it:
        out = np.dstack([blended_rgb, alpha * 255 * np.ones((h, w), dtype=np.uint8)])  # keep an explicit alpha layer set by opacity
    else:
        # RGB blend
        out_rgb = ((1.0 - alpha) * img_rgb + alpha * patch_resized).clip(0, 255).astype(np.uint8)
        out = out_rgb

    # If original was RGBA and we wanted to preserve original alpha channel, restore it
    if out_mode == 'RGBA' and has_alpha:
        # use original alpha channel (preserve transparency)
        out = np.dstack([out[..., :3], alpha * 255 * np.ones((h, w), dtype=np.uint8)])
        # prefer original alpha where it exists (non-zero) to avoid losing existing transparency
        out[..., 3] = alpha * 255 * (alpha > 0)  # keep simple consistent alpha

    return out.astype(np.uint8)


def add_attacks_to_folder(input_folder, output_folder, attack_types=None, patch_size=32, patch_intensity=140, opacity=0.35):
    """
    Add adversarial attacks to all images in a folder
    """
    if attack_types is None:
        attack_types = ['pgd', 'fgsm', 'cw', 'bpda', 'eot']

    input_path = Path(input_folder)
    if not input_path.exists():
        logging.error("Input folder does not exist: %s", input_folder)
        return False

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # create per-attack subfolders
    for at in attack_types:
        (output_path / at).mkdir(parents=True, exist_ok=True)

    # gather files
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    files = []
    for e in exts:
        files.extend(input_path.glob(e))
        files.extend(input_path.glob(e.upper()))

    if not files:
        logging.info("No images found in %s", input_folder)
        return False

    logging.info("Found %d images", len(files))
    results = {}

    for fp in files:
        try:
            img = Image.open(fp)
            if img.mode not in ('RGB', 'RGBA', 'L'):
                img = img.convert('RGB')
            base_name = fp.stem

            image_results = {}
            for at in attack_types:
                try:
                    attacked = _add_synthetic_patch(img, attack_type=at, size=patch_size,
                                                    intensity=patch_intensity,
                                                    seed=(hash(base_name + at) & 0xffffffff),
                                                    opacity=opacity)
                    # save as PNG to preserve mode/alpha
                    out_file = output_path / at / f"{base_name}.png"
                    Image.fromarray(attacked).save(out_file)
                    image_results[at] = {'output_path': str(out_file), 'success': True}
                    logging.info("Processed %s -> %s", fp.name, out_file.relative_to(output_path))
                except Exception as e:
                    logging.error("Failed to apply %s to %s: %s", at, fp.name, str(e))
                    image_results[at] = {'success': False, 'error': str(e)}
            results[base_name] = image_results
        except Exception as e:
            logging.error("Failed to load/process %s: %s", fp, str(e))
            traceback.print_exc()

    # summary
    total = 0
    ok = 0
    for name, res in results.items():
        for at, info in res.items():
            total += 1
            if info.get('success'):
                ok += 1

    logging.info("Finished: %d/%d succeeded", ok, total)
    return ok == total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add adversarial patch attacks to images in a folder")
    parser.add_argument("input_folder", nargs='?', default=r"C:\Users\paild\Desktop\Thesis Code\dataset")
    parser.add_argument("output_folder", nargs='?', default=r"C:\Users\paild\Desktop\Thesis Code\attacked_dataset")
    parser.add_argument("--attacks", default="pgd,fgsm,cw,bpda,eot",
                        help="Comma-separated attack types (pgd,fgsm,cw,bpda,eot)")
    parser.add_argument("--size", type=float, default=640.0,
                        help="Patch size in pixels (int) or relative fraction of min(image_dim) (float in (0,1])")
    parser.add_argument("--intensity", type=int, default=140, help="Patch intensity")
    parser.add_argument("--opacity", type=float, default=0.35, help="Overlay opacity (0.0-1.0), lower => more transparent")
    args = parser.parse_args(argv)

    attack_types = [a.strip() for a in args.attacks.split(',') if a.strip()]

    success = add_attacks_to_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        attack_types=attack_types,
        patch_size=args.size,
        patch_intensity=args.intensity,
        opacity=args.opacity
    )

    if success:
        logging.info("All attacks applied successfully. Output saved to %s", args.output_folder)
        return 0
    else:
        logging.warning("Some attacks failed. Check logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())