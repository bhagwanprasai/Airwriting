"""
Inference script for the CRNN handwritten word recognition model.

Usage:
    python infer.py <image_path>
    python infer.py <image_path> --model best_crnn_model.pth
    python infer.py <image_path> --model best_crnn_model.pth --encoder encoder_vocab.json

--model defaults to best_crnn_model.pth in the current directory.

The encoder vocab JSON can be generated from your training data by running:
    python infer.py --save-encoder

If no encoder file is provided, the script falls back to the full alphanumeric
character set (0-9, A-Z, a-z) which matches the training data filter.
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import shared components from model.py
# ---------------------------------------------------------------------------
from model import CRNN, LabelEncoder, Config


# ---------------------------------------------------------------------------
# Default vocabulary (matches clean_dataset regex: ^[a-zA-Z0-9]+$)
# Sorted to reproduce the same index mapping used during training.
# ---------------------------------------------------------------------------
DEFAULT_CHARS = sorted(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def build_default_encoder() -> LabelEncoder:
    """Build a LabelEncoder using the default alphanumeric character set."""
    enc = LabelEncoder()
    enc.char2idx = {ch: idx + 1 for idx, ch in enumerate(DEFAULT_CHARS)}
    enc.idx2char = {idx + 1: ch for idx, ch in enumerate(DEFAULT_CHARS)}
    enc.idx2char[0] = ""
    return enc


def load_encoder_from_json(path: str) -> LabelEncoder:
    """Load a LabelEncoder whose vocab was saved as a JSON mapping."""
    with open(path, "r") as f:
        data = json.load(f)
    enc = LabelEncoder()
    enc.char2idx = {k: int(v) for k, v in data["char2idx"].items()}
    enc.idx2char = {int(k): v for k, v in data["idx2char"].items()}
    return enc


def save_encoder_to_json(enc: LabelEncoder, path: str) -> None:
    """Persist a LabelEncoder vocab to a JSON file for later inference use."""
    data = {
        "char2idx": enc.char2idx,
        "idx2char": {str(k): v for k, v in enc.idx2char.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Encoder vocab saved to '{path}'")


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image the same way WordDataset does:
      - Read as grayscale
      - Resize height to Config.IMG_HEIGHT, keep aspect ratio
      - Clamp width to [MIN_WIDTH, MAX_WIDTH]
      - Normalise pixels to [0, 1]
      - Return shape (1, 1, H, W)  (batch=1, channel=1)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape
    target_w = int(w * (Config.IMG_HEIGHT / h))
    target_w = max(Config.MIN_WIDTH, min(target_w, Config.MAX_WIDTH))
    img = cv2.resize(img, (target_w, Config.IMG_HEIGHT))

    img = img.astype(np.float32) / 255.0
    tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def predict(image_path: str, model: CRNN, encoder: LabelEncoder) -> str:
    """Run inference on a single image and return the predicted word."""
    model.eval()
    with torch.no_grad():
        img_tensor = preprocess_image(image_path).to(Config.DEVICE)
        log_probs = model(img_tensor)                          # (T, 1, num_classes)
        pred_indices = torch.argmax(log_probs, dim=2)          # (T, 1)
        sequence = pred_indices[:, 0].cpu().numpy()
        return encoder.decode_greedy(sequence)


def load_model(encoder: LabelEncoder, model_path: str) -> CRNN:
    """Instantiate and load weights into the CRNN model."""
    model = CRNN(
        num_chars=encoder.num_classes(),
        hidden_size=Config.LSTM_HIDDEN,
        dropout=0.0,          # no dropout at inference
    ).to(Config.DEVICE)

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found at '{checkpoint}'.")

    model.load_state_dict(
        torch.load(str(checkpoint), map_location=Config.DEVICE, weights_only=True)
    )
    print(f"Loaded weights from '{checkpoint}'")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="CRNN inference: predict the word in a handwritten-word image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        metavar="PATH",
        help="Path to a saved encoder JSON (optional). "
             "Falls back to the default alphanumeric charset if omitted.",
    )
    parser.add_argument(
        "--model",
        default="best_crnn_model.pth",
        metavar="PATH",
        help="Path to the .pth model checkpoint (default: best_crnn_model.pth).",
    )
    parser.add_argument(
        "--save-encoder",
        metavar="PATH",
        default=None,
        const="encoder_vocab.json",
        nargs="?",
        help="Save the default encoder vocab to a JSON file and exit. "
             "Optionally provide a destination path (default: encoder_vocab.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --save-encoder mode: just write the vocab file and quit
    if args.save_encoder is not None:
        enc = build_default_encoder()
        save_encoder_to_json(enc, args.save_encoder)
        sys.exit(0)

    if args.image is None:
        print("Error: please supply an image path.\n")
        print("  python infer.py path/to/word.png")
        print("  python infer.py path/to/word.png --encoder encoder_vocab.json")
        sys.exit(1)

    # Build / load encoder
    if args.encoder and Path(args.encoder).exists():
        encoder = load_encoder_from_json(args.encoder)
        print(f"Loaded encoder from '{args.encoder}' ({len(encoder.char2idx)} chars).")
    else:
        if args.encoder:
            print(f"Warning: encoder file '{args.encoder}' not found, using default vocab.")
        encoder = build_default_encoder()
        print(f"Using default alphanumeric encoder ({len(encoder.char2idx)} chars).")

    model = load_model(encoder, args.model)

    prediction = predict(args.image, model, encoder)
    print(f"\nImage   : {args.image}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
