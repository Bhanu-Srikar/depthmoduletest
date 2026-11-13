"""
depth_module.py - monocular depth for the Team Codex landing-site pipeline.

Pipeline role (report, module 2 + part of module 4):
    image -> MiDaS_small -> normalized relative depth map
          -> roughness (slope proxy) map
          -> per-candidate-box roughness R and "too steep" check

Install:  pip install torch torchvision opencv-python numpy timm
First run needs internet (torch.hub downloads MiDaS code + weights, ~100 MB).

NOTE: MiDaS outputs RELATIVE INVERSE depth (bigger value = closer to camera),
with an unknown scale per image. So values are only comparable *within one image*,
and roughness/steepness thresholds must be tuned empirically, not in meters.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_transform = None


def _load():
    """Load MiDaS_small once and reuse it (avoid recomputation, per the risk plan)."""
    global _model, _transform
    if _model is None:
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        _model.to(_DEVICE).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _transform = transforms.small_transform
    return _model, _transform


@torch.no_grad()
def estimate_depth(image_rgb: np.ndarray) -> np.ndarray:
    """
    image_rgb: H x W x 3 uint8 array in RGB order (cv2 loads BGR - convert first!).
    Returns:   H x W float32 array in [0, 1] (min-max normalized, same size as input).
               Larger = closer (MiDaS inverse depth).
    """
    model, transform = _load()
    batch = transform(image_rgb).to(_DEVICE)
    pred = model(batch)
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    lo, hi = float(pred.min()), float(pred.max())
    return ((pred - lo) / (hi - lo + 1e-8)).astype(np.float32)


def roughness_map(depth: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """
    Per-pixel roughness / slope proxy = magnitude of the depth gradient.
    Flat ground -> ~0, sharp depth changes (walls, trees, edges) -> large.
    Light blur first so sensor/model noise is not counted as roughness.
    """
    d = cv2.GaussianBlur(depth, (blur_ksize, blur_ksize), 0)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def _box_values(rmap: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Roughness values inside a (possibly rotated) box given by 4 (x, y) corners."""
    corners = np.asarray(corners, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(corners)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, rmap.shape[1]), min(y + h, rmap.shape[0])
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [corners - np.array([x0, y0], dtype=np.int32)], 1)
    return rmap[y0:y1, x0:x1][mask.astype(bool)]


def box_roughness(rmap: np.ndarray, corners: np.ndarray) -> float:
    """R for the cost function: mean roughness inside the candidate box."""
    vals = _box_values(rmap, corners)
    return float(vals.mean()) if vals.size else float("inf")


def box_is_too_steep(rmap: np.ndarray, corners: np.ndarray,
                     threshold: float = 0.05, percentile: float = 95.0) -> bool:
    """
    Reject a candidate if its steepest parts exceed `threshold`.
    Uses a high percentile (not max) so one noisy pixel does not kill a good box.
    `threshold` is a placeholder - TUNE it on several dataset images.
    """
    vals = _box_values(rmap, corners)
    if vals.size == 0:
        return True
    return bool(np.percentile(vals, percentile) > threshold)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    bgr = cv2.imread(path)
    assert bgr is not None, f"could not read {path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = estimate_depth(rgb)
    rmap = roughness_map(depth)

    def to_color(a):
        a = (255 * (a - a.min()) / (a.max() - a.min() + 1e-8)).astype(np.uint8)
        return cv2.applyColorMap(a, cv2.COLORMAP_INFERNO)

    cv2.imwrite("depth_vis.png", to_color(depth))
    cv2.imwrite("roughness_vis.png", to_color(rmap))
    print("depth range:", depth.min(), depth.max())
    print("roughness mean/95th pct:", rmap.mean(), np.percentile(rmap, 95))

    # example: test a 100x100 box near the image centre
    h, w = depth.shape
    cx, cy, s = w // 2, h // 2, 50
    box = np.array([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]])
    print("example box R:", box_roughness(rmap, box),
          "| too steep:", box_is_too_steep(rmap, box))
