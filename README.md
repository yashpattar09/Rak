# Seed2 Matcher

Instance-level image matcher for ground/site photos. Given a set of **reference**
images (specific arrangements of debris — white shards, rocks, bricks — on dirt)
and a set of **query** images (wider / different-angle / different-lighting shots
of the same ground), it decides which query corresponds to each reference and
tells you **how confident** that match is.

It runs as a small **Flask web app**: open it in a browser, upload a ZIP of query
images, click **Start Matching**, and get a side-by-side pairing with a confidence
badge per match.

---

## Why this is hard (and how it's solved)

Almost every image is "brown dirt." A single small global embedding can barely
tell two scenes apart, and matches are the **same physical spot** photographed
from a different angle and under different lighting. So the matcher fuses three
complementary signals instead of relying on one:

| # | Signal | What it contributes |
|---|--------|---------------------|
| 1 | **ResNet18 @ 128×128** (core, unchanged) | Fast global embedding; the original baseline. |
| 2 | **DINOv2 (ViT-S/14)** full-res, multi-scale + flip TTA | A far more discriminative global descriptor for ranking near-identical scenes. |
| 3 | **SIFT + RANSAC geometric verification** full-res | Counts geometrically-consistent keypoint correspondences — direct physical evidence that two frames show the same surface, and the **confidence driver**. |

The three score matrices are min-max normalised and fused:

```
fused = 0.15 · ResNet18  +  0.35 · DINOv2  +  0.50 · geometry
```

Geometry gets the highest weight because RANSAC inliers are the closest thing to
ground truth here. A one-to-one **Hungarian assignment** then runs on the fused
matrix (each reference maps to a distinct query).

### Confidence

A match is marked **confident** when it has **≥ 15 RANSAC-verified inliers**
(`CONF_INLIERS`). This means weak/uncertain pairings are honestly flagged instead
of being shown with false confidence. When geometry is unavailable, confidence
falls back to a cosine-similarity threshold (`THRESH = 0.55`).

### Graceful degradation

If DINOv2 or SIFT can't load (e.g. offline, or the GPU is out of memory), those
signals switch off automatically and the pipeline degrades to the **original
ResNet18 @ 128×128 behaviour**. DINOv2 additionally falls back from GPU to CPU if
CUDA allocation fails (useful on memory-constrained devices such as Jetson).

---

## Pipeline at a glance

```
                 ┌─────────────────────────── per image ───────────────────────────┐
  image ─┬─ resize 128×128 ─→ ResNet18 ─→ 512-d embedding ────┐
         │                                                     │
         ├─ full-res RGB ───→ DINOv2 (multi-scale + flip) ─────┤
         │                                                     │
         └─ full-res gray ──→ SIFT keypoints/descriptors ──────┘
                                                               │
   refs × queries  ─→  cosine (ResNet) ┐                       │
                   ─→  cosine (DINOv2) ├─ min-max norm ─→ weighted fuse ─→ Hungarian
                   ─→  RANSAC inliers  ┘                                        │
                                                                               ▼
                                                        one-to-one matches + confidence
```

---

## Requirements

- **Python 3.10+**
- Python packages: `flask`, `numpy`, `scipy`, `opencv-python` (4.4+, ships SIFT),
  `torch`, `torchvision`
- **Internet on first run** — pretrained weights download once and are cached:
  - ResNet18 (torchvision)
  - DINOv2 `dinov2_vits14` (via `torch.hub`, `facebookresearch/dinov2`)
- **GPU optional.** CUDA is used automatically if available; otherwise everything
  runs on CPU (slower, still works).

## Installation

```bash
# from the project root
pip install flask numpy scipy opencv-python torch torchvision
```

> On most setups `torch` / `torchvision` / `opencv` are already present; in
> practice `flask` is often the only missing package.

## Running

```bash
python3 dashboard_final.py
```

On startup it prints the device, loads the models, and serves the UI at:

```
http://localhost:5000
```

The first launch downloads the DINOv2 weights (~85 MB); subsequent launches use
the cache.

## Using the web UI

1. Open <http://localhost:5000>.
2. **📁 Upload ZIP** — pick a `.zip` of query images (`.jpg/.jpeg/.png/.bmp`).
   Skip this to match the built-in default query folder (`newwww/`).
3. **▶ Start Matching**.
4. Read the results. Each card shows:
   - the **reference** ↔ **matched query** pair,
   - `similarity% · geo N` — DINOv2 visual similarity and the number of
     RANSAC-verified keypoint correspondences,
   - a **confident / weak** badge.

---

## Configuration

All knobs live near the top of `dashboard_final.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `REF_FOLDER` | `extracted_seed2/captures1` | Reference image set. |
| `QUERY_FOLDER` | `newwww` | Default query set (used when no ZIP is uploaded). |
| `OUT_FOLDER` | `results_128x128` | Where 128×128 display crops are written. |
| `DEVICE` | auto | `cuda` if available, else `cpu`. |
| `USE_DINO` | `True` | Enable the DINOv2 signal. |
| `USE_GEOM` | `True` | Enable SIFT + RANSAC geometric verification. |
| `W_CORE, W_DINO, W_GEOM` | `0.15, 0.35, 0.50` | Fusion weights (renormalised over whichever signals load). |
| `GEOM_MAXSIDE` | `1600` | Longest image side used for SIFT (accuracy vs speed). |
| `GEOM_CAP` | `40` | Inlier count is clipped here before normalising. |
| `CONF_INLIERS` | `15` | Inliers needed to call a match **confident**. |
| `THRESH` | `0.55` | Cosine confidence threshold (fallback modes). |

To reproduce the original single-model behaviour, set `USE_DINO = False` and
`USE_GEOM = False`.

## HTTP API

The UI is backed by three endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/upload` | Upload a `.zip` of query images (multipart field `file`). Extracts and sets it as the active query set. |
| `POST` | `/api/match` | Run matching against the current query set; returns JSON results. |
| `GET`  | `/img/<path>` | Serve a 128×128 display crop. |

`/api/match` returns:

```json
{
  "results": [
    {
      "reference": "IMG_...HDR.jpg",
      "ref_path": "results_128x128/ref/IMG_...jpg",
      "query": "WhatsApp Image ....jpg",
      "query_path": "results_128x128/query/....jpg",
      "similarity": 66,
      "geo": 32,
      "confident": true,
      "top3": [ { "name": "...", "path": "...", "sim": 66, "geo": 32 } ]
    }
  ]
}
```

## Project structure

```
.
├── dashboard_final.py         # the whole app: models, matching, Flask UI
├── extracted_seed2/
│   └── captures1/captures/    # reference images
├── newwww/                    # default query images
└── results_128x128/
    ├── ref/                   # 128×128 reference display crops (generated)
    └── query/                 # 128×128 query display crops (generated)
```

---

## Interpreting results (important)

- **`geo` is the trustworthy number.** A high inlier count (e.g. 30–50+) is strong
  evidence of a real same-location match; single digits mean "not verified."
- **Geometry locks onto the ground plane.** Because the ground is roughly planar,
  RANSAC can confirm *same physical location* even if the loose debris was moved
  or the framing differs — which is why a correct match can show a **low DINOv2
  visual %** but a **high `geo`**. If your goal is matching the *specific object
  arrangement* rather than the *location*, the geometry weighting should be
  revisited.
- **Not every reference has a true match.** Some reference scenes were physically
  rearranged or simply aren't present in the query set; those are correctly
  surfaced as **weak**.

## Troubleshooting

- **`ModuleNotFoundError: flask`** → `pip install flask`.
- **DINOv2 fails to download** → the app logs a warning and continues without it
  (ResNet18 + geometry only). Re-run when back online to cache the weights.
- **GPU out-of-memory on startup** → DINOv2 automatically retries on CPU; matching
  still runs, just slower.
- **No matches / "No images found"** → check that the query folder or uploaded ZIP
  actually contains `.jpg/.jpeg/.png/.bmp` files.
