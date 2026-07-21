"""
plant image -> image_int.dat (FPGA input format)

usage:
  python prepare_image.py <image_path>

example:
  python prepare_image.py "D:/path/to/strawberry_test.jpg"

format reproduces uart_image.py logic:
  load -> RGB -> resize(256) -> center crop 224 -> normalize (ImageNet mean/std) -> *2^7 -> int
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_QUANT   = 7   # FPGA expects image * 2^7 then int

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / 'image_int.dat'


def preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')

    # Resize so smaller side = 256
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(round(h * 256 / w))
    else:
        new_w, new_h = int(round(w * 256 / h)), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224x224
    left = (new_w - 224) // 2
    top  = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                      # CHW
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

    arr = arr * (2 ** IMAGE_QUANT)
    arr = arr.astype(np.int32)
    return arr   # [3, 224, 224]


def write_dat(arr: np.ndarray, out_path: Path) -> None:
    with open(out_path, 'w') as f:
        flat = arr.flatten()
        f.write(' '.join(str(int(v)) for v in flat))
        f.write(' ')


def main():
    if len(sys.argv) != 2:
        print("usage: python prepare_image.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not Path(img_path).exists():
        print(f"[ERROR] not found: {img_path}")
        sys.exit(1)

    print(f"Input image: {img_path}")
    arr = preprocess(img_path)
    print(f"Array      : shape={tuple(arr.shape)}, dtype={arr.dtype}")
    print(f"Range      : [{arr.min()}, {arr.max()}]")
    print(f"Sample     : {arr.flatten()[:12].tolist()}")

    write_dat(arr, OUT_PATH)
    sz = OUT_PATH.stat().st_size
    print(f"\nWrote: {OUT_PATH}")
    print(f"  size: {sz:,} bytes")


if __name__ == '__main__':
    main()
