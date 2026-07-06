#!/usr/bin/env python3
"""Export a MobileNetV3-Small feature extractor to ONNX for deltax2sort.

Run this ONCE on your workstation to produce ``models/embedder.onnx``. The Rust
side (phase B2) loads that file with `tract` and feeds it object crops; the
resulting embedding vectors are stored, per labelled example, in the object
catalogue (the "learned file"). Recognition is then nearest-neighbour cosine
similarity over those embeddings — no on-device training.

The embedder is a *pretrained* network with its classifier head removed: it is
never trained here. You only ever teach the system by adding labelled example
crops to the catalogue, which is why a single generic embedder works.

Usage:
    pip install torch torchvision onnx
    python models/export_embedder.py            # -> models/embedder.onnx

CONTRACT (the Rust embedder must match this exactly):
  * Input : 1x3xHxW float32, RGB, H=W=224.
  * Preprocessing: resize the crop to 224x224, convert to RGB, scale to [0,1]
    (divide by 255), then normalise with ImageNet mean/std:
        mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    channel order CHW.
  * Output: 1x576 float32 embedding (MobileNetV3-Small pooled features).
    The Rust side L2-normalises it before storing / comparing.
"""

import argparse
import pathlib

import torch
import torchvision


INPUT_SIZE = 224
EMBED_DIM = 576  # MobileNetV3-Small pooled feature width


def build_embedder() -> torch.nn.Module:
    """Pretrained MobileNetV3-Small with the classifier replaced by a flatten,
    so the network outputs the 576-d pooled feature vector (the embedding)."""
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    net = torchvision.models.mobilenet_v3_small(weights=weights)
    # features -> avgpool -> flatten. Drop the classifier head.
    net.classifier = torch.nn.Flatten()
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out",
        default=str(pathlib.Path(__file__).with_name("embedder.onnx")),
        help="output ONNX path (default: models/embedder.onnx)",
    )
    args = parser.parse_args()

    net = build_embedder()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, EMBED_DIM), f"unexpected embedding shape {tuple(out.shape)}"

    torch.onnx.export(
        net,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=13,
        dynamic_axes=None,  # fixed 1x3x224x224 — simplest for tract
    )
    print(f"wrote {args.out}  (input 1x3x{INPUT_SIZE}x{INPUT_SIZE} f32 -> embedding 1x{EMBED_DIM})")


if __name__ == "__main__":
    main()
