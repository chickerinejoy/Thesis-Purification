from purification import AdversarialPatchPurifier

def main():
    purifier = AdversarialPatchPurifier()
    img = purifier.load_image(r"c:\Users\paild\Desktop\Thesis Code\adversarial_image.jpeg")

    purified, mask = purifier.purify_image(
        img,
        attack_type='auto',
        iterations=3,
        super_resolve=True,
        sr_scale=4,  # use 2 if you downloaded an x2 model
        sr_model_path=r"c:\Users\paild\Desktop\Thesis Code\models\RealESRGAN_x4plus.pth",
        sr_device='cpu'  # or 'cuda' if you installed GPU torch
    )

    purifier.save_image(purified, r"c:\Users\paild\Desktop\Thesis Code\purified_knife_sr.png")
    print("Saved purified + SR image as purified_knife_sr.png")
    print(f"Detected patch pixels: {int(mask.sum())}")

if __name__ == "__main__":
    main()