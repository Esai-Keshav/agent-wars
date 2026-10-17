"""Generate a sample loading challan image for the multimodal demo (spec 7)."""
from PIL import Image, ImageDraw

import config


def make_sample(path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "LOADING CHALLAN", fill="black")
    d.text((20, 60), "Sender: Sathya Traders Chennai", fill="black")
    d.text((20, 100), "Pallets: 6", fill="black")
    d.text((20, 140), "Weight: 1500 kg", fill="black")
    d.text((20, 180), "Destination: Sriperumbudur", fill="black")
    img.save(path, "JPEG")


if __name__ == "__main__":
    make_sample(config.SAMPLE_RECEIPT)
    print("wrote", config.SAMPLE_RECEIPT)
