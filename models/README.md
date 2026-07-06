# Recognition model

The object recogniser (issue #47, phase B) uses a **pretrained image embedder**
plus nearest-neighbour matching over a **learned catalogue**. The embedder is
never trained here — you teach the system only by adding labelled example crops
to the catalogue on the workstation.

## Producing the embedder

```bash
pip install torch torchvision onnx
python models/export_embedder.py     # writes models/embedder.onnx
```

`embedder.onnx` is a binary (~5–10 MB) and is **not** committed — generate it
once per machine (or copy it alongside the learned catalogue to the Pi).

## Contract

The Rust embedder (`src/vision/embedder.rs`, phase B2) must feed the model
exactly what the export script documents:

- input `1x3x224x224` float32, **RGB**, CHW;
- resize crop → 224×224, scale to `[0,1]`, normalise with ImageNet
  `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`;
- output `1x576` float32 embedding, L2-normalised before use.

## Files

- `export_embedder.py` — one-shot ONNX exporter (committed).
- `embedder.onnx` — the model (git-ignored; you generate it).
- The learned catalogue (labelled embeddings) lives wherever
  `[recognition].catalog_path` points, **not** here.
