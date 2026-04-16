import json
import math
import os
import tempfile

import cv2
import folder_paths
import numpy as np
import requests
import torch
from scipy.signal import find_peaks

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif")


def _download_if_url(path):
    """If path is a URL, download to temp dir. Returns (local_path, is_temp)."""
    if path.startswith("http"):
        tmp_dir = folder_paths.get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", dir=tmp_dir, delete=False)
        resp = requests.get(path, stream=True, timeout=120)
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()
        return tmp.name, True
    return path, False


def _make_thumbnails(frames, width=160):
    """Downscale batch tensor [N,H,W,C] to grayscale thumbnails for fast comparison."""
    n, h, w, c = frames.shape
    scale = width / w
    new_h = max(1, int(h * scale))
    # Convert to uint8 numpy for cv2
    arr = (frames.numpy() * 255).astype(np.uint8)
    thumbs = []
    for i in range(n):
        small = cv2.resize(arr[i], (width, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) if small.ndim == 3 else small
        thumbs.append(gray)
    return thumbs


def _motion_signal(thumbs):
    """Compute per-frame motion magnitude (mean diff from previous frame)."""
    signal = [0.0]
    for i in range(1, len(thumbs)):
        signal.append(float(np.mean(cv2.absdiff(thumbs[i], thumbs[i - 1]))))
    return np.array(signal)


def _compute_masks(thumbs):
    """Extract character masks from thumbnails using background detection + Otsu."""
    masks = []
    for thumb in thumbs:
        h, w = thumb.shape[:2]
        # Sample edges as background
        edge_pixels = np.concatenate([
            thumb[:4, :].flatten(),
            thumb[-4:, :].flatten(),
            thumb[:, :4].flatten(),
            thumb[:, -4:].flatten(),
        ])
        bg_val = np.median(edge_pixels)
        diff = np.abs(thumb.astype(np.float32) - bg_val)
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        masks.append(mask)
    return masks


def _extract_leg_signal(thumbs, masks):
    """Extract leg spread signal from bottom portion of character silhouette.

    Returns smoothed signal where peaks = legs spread (mid-stride),
    valleys = legs together (passing position).
    """
    n = len(thumbs)
    leg_spread = np.zeros(n)

    for i in range(n):
        mask = masks[i]
        h, w = mask.shape
        ys, xs = np.where(mask > 127)
        if len(ys) < 10:
            continue

        # Character bounding box
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        char_h = y_max - y_min
        if char_h < 5:
            continue

        # Bottom 40% of the character = legs
        leg_top = y_min + int(char_h * 0.6)
        leg_mask = mask[leg_top:y_max + 1, x_min:x_max + 1]
        if leg_mask.shape[0] < 2 or leg_mask.shape[1] < 2:
            continue

        leg_h, leg_w = leg_mask.shape
        mid_x = leg_w // 2

        # Left and right leg regions
        left_ys, _ = np.where(leg_mask[:, :mid_x] > 127)
        right_ys, _ = np.where(leg_mask[:, mid_x:] > 127)

        left_y = float(np.mean(left_ys)) if len(left_ys) > 0 else leg_h / 2
        right_y = float(np.mean(right_ys)) if len(right_ys) > 0 else leg_h / 2

        leg_spread[i] = abs(left_y - right_y)

    # Smooth with Gaussian kernel
    if n > 5:
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, 1.5).flatten()
        leg_spread = np.convolve(leg_spread, kernel, mode='same')

    return leg_spread


def _boundary_score(thumbs, i, j, window=3):
    """Multi-frame boundary match score (lower = better match)."""
    n = len(thumbs)
    total, count = 0.0, 0
    for k in range(-window, window + 1):
        fi, fj = i + k, j + k
        if 0 <= fi < n and 0 <= fj < n:
            total += float(np.mean(cv2.absdiff(thumbs[fi], thumbs[fj])))
            count += 1
    return total / count if count > 0 else 999.0


def _score_cycle(start, end, thumbs, motion, leg_spread, num_frames):
    """Score a candidate cycle. Lower = better."""
    n = len(thumbs)
    cycle_len = end - start

    # A. Boundary match (weight 5)
    boundary = _boundary_score(thumbs, start, end)

    # B. Half-cycle symmetry (weight 2)
    # Find the mid-peak between start and end
    mid = start + cycle_len // 2
    half1 = mid - start
    half2 = end - mid
    symmetry = abs(half1 - half2) / max(half1, half2, 1)

    # C. Spacing fitness for num_frames (weight 1.5)
    step = cycle_len / num_frames
    frac = step - math.floor(step)
    spacing = min(frac, 1 - frac)  # 0 = perfect, 0.5 = worst

    # D. Motion quality (weight 1)
    cycle_motion = motion[start:end]
    if len(cycle_motion) > 2:
        mean_m = float(np.mean(cycle_motion[1:]))  # skip first (always 0)
        stall_count = np.sum(cycle_motion[1:] < mean_m * 0.2) if mean_m > 0 else 0
        stall_pct = stall_count / len(cycle_motion)
    else:
        stall_pct = 0.0

    score = boundary * 5.0 + symmetry * 2.0 + spacing * 1.5 + stall_pct * 1.0
    return score


def _find_best_cycle(thumbs, min_len, max_len, num_frames=8):
    """Find the best walk cycle using leg silhouette analysis.

    1. Extract character masks and leg spread signal
    2. Find stride peaks (mid-stride positions)
    3. Enumerate peak-to-peak+2 candidate cycles
    4. Score each, pick best
    5. Fallback to sliding window if peak detection fails
    """
    n = len(thumbs)
    motion = _motion_signal(thumbs)
    masks = _compute_masks(thumbs)
    leg_spread = _extract_leg_signal(thumbs, masks)

    print(f"[LoopStrip] Leg signal range: {leg_spread.min():.1f}-{leg_spread.max():.1f}, "
          f"std={leg_spread.std():.2f}")

    # Find stride peaks
    min_peak_distance = max(3, min_len // 3)
    prominence = max(0.5, leg_spread.std() * 0.5)
    peaks, properties = find_peaks(leg_spread, distance=min_peak_distance, prominence=prominence)

    print(f"[LoopStrip] Found {len(peaks)} stride peaks at frames: {peaks.tolist()}")

    # Enumerate candidate cycles: peak[i] to peak[i+2] = one full walk cycle
    candidates = []
    for i in range(len(peaks) - 2):
        start = int(peaks[i])
        end = int(peaks[i + 2])
        cycle_len = end - start
        if min_len <= cycle_len <= max_len:
            score = _score_cycle(start, end, thumbs, motion, leg_spread, num_frames)
            candidates.append((start, end, score))

    # Also try peak[i] to peak[i+1] (half cycles) if they're long enough
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        cycle_len = end - start
        if min_len <= cycle_len <= max_len:
            score = _score_cycle(start, end, thumbs, motion, leg_spread, num_frames)
            candidates.append((start, end, score))

    if candidates:
        candidates.sort(key=lambda x: x[2])
        best = candidates[0]
        print(f"[LoopStrip] Best cycle from peaks: {best[0]}-{best[1]} "
              f"(len={best[1]-best[0]}, score={best[2]:.2f}, "
              f"from {len(candidates)} candidates)")
        return best[0], best[1], best[2]

    # ── Fallback: sliding window with multiple period candidates ──
    print(f"[LoopStrip] Peak detection insufficient, falling back to sliding window")

    # Try multiple window sizes
    best_score = float("inf")
    best_i, best_j = 0, min_len

    for win_len in range(min_len, max_len + 1):
        for i in range(0, n - win_len + 1):
            j = i + win_len
            score = _score_cycle(i, j, thumbs, motion, leg_spread, num_frames)
            if score < best_score:
                best_score = score
                best_i, best_j = i, j

    print(f"[LoopStrip] Fallback best: {best_i}-{best_j} "
          f"(len={best_j-best_i}, score={best_score:.2f})")
    return best_i, best_j, float(best_score)


# ─── Load Video ─────────────────────────────────────────────────────────────


class LoopStripLoadVideo:
    """Load video frames from upload or file path / URL."""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTS)])
        return {
            "required": {
                "video": (files, {"video_upload": True}),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Override: file path or URL (takes priority over upload)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("frames", "fps", "frame_count")
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"

    def execute(self, video, video_path=""):
        # Determine source
        if video_path and video_path.strip():
            path = video_path.strip()
        else:
            path = os.path.join(folder_paths.get_input_directory(), video)

        local_path, is_temp = _download_if_url(path)

        try:
            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {local_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(rgb).float() / 255.0)

            cap.release()
        finally:
            if is_temp and os.path.exists(local_path):
                os.unlink(local_path)

        if not frames:
            raise ValueError("No frames could be read from video")

        batch = torch.stack(frames)  # [N, H, W, C]
        print(f"[LoopStrip] Loaded {len(frames)} frames, {fps:.1f} fps, {batch.shape[2]}x{batch.shape[1]}")
        return (batch, fps, len(frames))


# ─── Find Cycle ─────────────────────────────────────────────────────────────


class LoopStripFindCycle:
    """Find the best seamless loop cycle in a batch of frames."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "num_frames": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of frames to extract from the loop cycle",
                }),
                "min_cycle_pct": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Minimum cycle length as fraction of total frames",
                }),
                "max_cycle_pct": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.3,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Maximum cycle length as fraction of total frames",
                }),
                "grid_cols": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Grid columns (1 = no grid split)"}),
                "grid_rows": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Grid rows (1 = no grid split)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("loop_frames", "grid_1", "grid_2", "grid_3", "grid_4")
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"

    def _find_cycle_indices(self, frames, num_frames, min_cycle_pct, max_cycle_pct, label=""):
        """Find best cycle and return the frame indices to use."""
        total = frames.shape[0]
        if total < 3:
            raise ValueError(f"Need at least 3 frames, got {total}")

        min_len = max(2, int(total * min_cycle_pct))
        max_len = min(total - 1, int(total * max_cycle_pct))
        if min_len >= max_len:
            min_len = 2
            max_len = total - 1

        print(f"[LoopStrip] {label}Searching for cycle in {total} frames (len {min_len}-{max_len})...")

        thumbs = _make_thumbnails(frames)
        cycle_start, cycle_end, score = _find_best_cycle(thumbs, min_len, max_len, num_frames)
        cycle_len = cycle_end - cycle_start
        print(f"[LoopStrip] {label}Best cycle: frames {cycle_start}-{cycle_end} "
              f"(len={cycle_len}, score={score:.2f})")

        # Refine cycle end
        best_end = cycle_end
        best_boundary = _boundary_score(thumbs, cycle_start, cycle_end, window=2)
        for offset in range(-5, 6):
            candidate_end = cycle_end + offset
            if candidate_end > cycle_start + 4 and candidate_end < total:
                b = _boundary_score(thumbs, cycle_start, candidate_end, window=2)
                if b < best_boundary:
                    best_boundary = b
                    best_end = candidate_end
        if best_end != cycle_end:
            cycle_end = best_end
            cycle_len = cycle_end - cycle_start

        cycle_thumbs = thumbs[cycle_start:cycle_end]
        best_offset = 0
        best_loop_score = float("inf")

        for offset in range(cycle_len):
            indices = [(offset + i * cycle_len // num_frames) % cycle_len for i in range(num_frames)]
            last_thumb = cycle_thumbs[indices[-1]]
            first_thumb = cycle_thumbs[indices[0]]
            loop_score = float(np.mean(cv2.absdiff(last_thumb, first_thumb)))
            max_jump = 0.0
            for k in range(len(indices) - 1):
                jump = float(np.mean(cv2.absdiff(cycle_thumbs[indices[k]], cycle_thumbs[indices[k+1]])))
                max_jump = max(max_jump, jump)
            loop_penalty = max(0, loop_score - max_jump * 1.2)
            total_score = loop_score + loop_penalty * 3.0
            if total_score < best_loop_score:
                best_loop_score = total_score
                best_offset = offset

        indices = [cycle_start + ((best_offset + i * cycle_len // num_frames) % cycle_len)
                   for i in range(num_frames)]
        print(f"[LoopStrip] {label}offset={best_offset}, absolute indices={indices}")
        return indices

    def execute(self, frames, num_frames=8, min_cycle_pct=0.15, max_cycle_pct=0.8,
                grid_cols=1, grid_rows=1):
        n, h, w, c = frames.shape

        # Find cycle on the FULL video — same indices apply to all grid cells (synced)
        indices = self._find_cycle_indices(frames, num_frames, min_cycle_pct, max_cycle_pct)
        loop_frames = frames[indices]

        # Grid outputs: same indices, different spatial cells
        grid_outputs = []
        if grid_cols > 1 or grid_rows > 1:
            cell_h = h // grid_rows
            cell_w = w // grid_cols
            for r in range(grid_rows):
                for col_i in range(grid_cols):
                    y1 = r * cell_h
                    x1 = col_i * cell_w
                    cell = frames[:, y1:y1 + cell_h, x1:x1 + cell_w, :]
                    grid_outputs.append(cell[indices])

        # Pad grid outputs to 4
        empty = torch.zeros(1, 1, 1, c)
        while len(grid_outputs) < 4:
            grid_outputs.append(empty)

        return (loop_frames, grid_outputs[0], grid_outputs[1], grid_outputs[2], grid_outputs[3])


# ─── Center Subject ─────────────────────────────────────────────────────────


class LoopStripCenterSubject:
    """Center the subject in every frame. Auto-detects by sampling background color from corners."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output frame width",
                }),
                "output_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output frame height",
                }),
                "fill_percent": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.30,
                    "max": 0.95,
                    "step": 0.05,
                    "round": 0.01,
                    "tooltip": "How much of the canvas the character fills (0.6 = 60%, the blue box)",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.12,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "How different a pixel must be from the background to count as subject",
                }),
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": "Optional masks from segmentation node. If not provided, auto-detects via background color.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("centered_frames",)
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"

    def _detect_background(self, frames):
        """Sample corners and edges of all frames to estimate background color."""
        n, h, w, c = frames.shape
        samples = []
        # Sample 8px strips from all 4 edges across all frames
        s = 8
        samples.append(frames[:, :s, :, :].reshape(-1, c))        # top
        samples.append(frames[:, -s:, :, :].reshape(-1, c))       # bottom
        samples.append(frames[:, :, :s, :].reshape(-1, c))        # left
        samples.append(frames[:, :, -s:, :].reshape(-1, c))       # right
        all_samples = torch.cat(samples, dim=0)  # [M, C]
        # Median is robust to a character partially overlapping edges
        bg_color = torch.median(all_samples, dim=0).values  # [C]
        return bg_color

    def execute(self, frames, output_width=512, output_height=512, fill_percent=0.6,
                threshold=0.12, masks=None):
        n, h, w, c = frames.shape
        bg_color = self._detect_background(frames)

        if masks is not None:
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            if masks.shape[0] == 1 and n > 1:
                masks = masks.repeat(n, 1, 1)
        else:
            # Auto-detect via Otsu
            diffs = (frames - bg_color.view(1, 1, 1, c)).abs().mean(dim=-1)
            masks = torch.zeros_like(diffs)
            kernel = np.ones((5, 5), np.uint8)
            for i in range(n):
                d = (diffs[i].numpy() * 255).clip(0, 255).astype(np.uint8)
                _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
                masks[i] = torch.from_numpy(m).float() / 255.0

        # Find bounding box per frame
        bboxes = []
        for i in range(n):
            mask = masks[i] if i < masks.shape[0] else masks[-1]
            ys, xs = torch.where(mask > 0.5)
            if len(ys) == 0:
                bboxes.append((0, 0, w, h))
            else:
                bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

        # Union bbox across all frames (stable, no jitter)
        union_x1 = min(b[0] for b in bboxes)
        union_y1 = min(b[1] for b in bboxes)
        union_x2 = max(b[2] for b in bboxes)
        union_y2 = max(b[3] for b in bboxes)
        subj_h = union_y2 - union_y1
        subj_w = union_x2 - union_x1

        # Median of all mask pixels across all frames = robust center
        all_xs, all_ys = [], []
        for i in range(n):
            mask = masks[i] if i < masks.shape[0] else masks[-1]
            ys, xs = torch.where(mask > 0.5)
            if len(xs) > 0:
                all_xs.append(xs)
                all_ys.append(ys)
        if all_xs:
            face_cx = float(torch.median(torch.cat(all_xs).float()))
            face_cy = float(torch.median(torch.cat(all_ys).float()))
        else:
            face_cx = (union_x1 + union_x2) / 2
            face_cy = (union_y1 + union_y2) / 2

        print(f"[LoopStrip] Center Subject median=({face_cx:.0f},{face_cy:.0f})")

        target_char_h = output_height * fill_percent
        scale = target_char_h / subj_h if subj_h > 0 else 1.0

        # Check if width also fits
        if subj_w * scale > output_width * fill_percent:
            scale = (output_width * fill_percent) / subj_w

        # Scale the entire frame
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        scaled = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2), size=(new_h, new_w),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

        # Face center in scaled coordinates
        face_cx_s = int(face_cx * scale)
        face_cy_s = int(face_cy * scale)

        # Canvas center
        canvas_cx = output_width // 2
        canvas_cy = output_height // 2

        # Copy region: place face center at canvas center
        src_x1 = face_cx_s - canvas_cx
        src_y1 = face_cy_s - canvas_cy
        src_x2 = src_x1 + output_width
        src_y2 = src_y1 + output_height

        # Clamp and compute offsets
        dst_x1 = max(0, -src_x1)
        dst_y1 = max(0, -src_y1)
        src_x1 = max(0, src_x1)
        src_y1 = max(0, src_y1)
        src_x2 = min(new_w, src_x2)
        src_y2 = min(new_h, src_y2)
        copy_w = src_x2 - src_x1
        copy_h = src_y2 - src_y1

        # White canvas (or bg_color), paste
        canvas = bg_color.view(1, 1, 1, c).expand(n, output_height, output_width, c).clone()
        if copy_w > 0 and copy_h > 0:
            canvas[:, dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w, :] = \
                scaled[:, src_y1:src_y2, src_x1:src_x2, :]

        print(f"[LoopStrip] Face-centered: face=({face_cx:.0f},{face_cy:.0f}) "
              f"bbox=({union_x1},{union_y1})-({union_x2},{union_y2}) scale={scale:.2f}")

        return (canvas,)


# ─── Assemble Strip ─────────────────────────────────────────────────────────


class LoopStripAssemble:
    """Assemble frames into a horizontal or vertical sprite strip."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("strip",)
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"
    OUTPUT_NODE = True

    def execute(self, frames, direction="horizontal"):
        # frames: [N, H, W, C]
        frame_list = [frames[i] for i in range(frames.shape[0])]
        if direction == "horizontal":
            strip = torch.cat(frame_list, dim=1)  # cat along W
        else:
            strip = torch.cat(frame_list, dim=0)  # cat along H

        strip = strip.unsqueeze(0)  # [1, H, W*N, C] or [1, H*N, W, C]
        print(f"[LoopStrip] Strip assembled: {strip.shape[2]}x{strip.shape[1]} ({direction})")
        return (strip,)


# ─── Center Character (single image) ────────────────────────────────────────


class LoopStripCenterCharacter:
    """Cut character using mask, scale to uniform height, paste centered on white.

    Pipeline: [Load Image] + [BiRefNet/RMBG] → masks → [Center Character]
    Auto-detects from background if no mask provided.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output square size"}),
                "fill_percent": ("FLOAT", {"default": 0.55, "min": 0.3, "max": 0.95, "step": 0.05,
                    "round": 0.01,
                    "tooltip": "How much of the canvas the character fills"}),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Character mask from BiRefNet/RMBG. If not provided, auto-detects from background.",
                }),
                "threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Auto-detection threshold (only used when no mask provided)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("centered",)
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"

    def execute(self, image, output_size=512, fill_percent=0.55,
                mask=None, threshold=0.12):
        n, h, w, c = image.shape

        # Strip alpha if present (RGBA → RGB)
        if c == 4:
            alpha = image[:, :, :, 3:4]
            image = image[:, :, :, :3]
            if mask is None:
                alpha_flat = alpha.squeeze(-1)
                if (alpha_flat < 0.5).any():
                    mask = alpha_flat
            c = 3

        # Prepare masks
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and n > 1:
                mask = mask.repeat(n, 1, 1)
            masks_list = [mask[min(i, mask.shape[0] - 1)] for i in range(n)]
            use_mask = True
        else:
            masks_list = [None] * n
            use_mask = False

        target = int(output_size * fill_percent)
        cascade_path = os.path.join(os.path.dirname(__file__), 'lbpcascade_animeface.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # ── Pass 1: detect masks, crops, and faces ──
        infos = []  # (frame, cropped, ch, cw, face_cx, face_cy, method) or None
        for i in range(n):
            frame = image[i]

            if use_mask:
                char_mask = masks_list[i] > 0.5
            else:
                samples = [
                    frame[:8, :, :].reshape(-1, c),
                    frame[-8:, :, :].reshape(-1, c),
                    frame[:, :8, :].reshape(-1, c),
                    frame[:, -8:, :].reshape(-1, c),
                ]
                bg = torch.median(torch.cat(samples, dim=0), dim=0).values
                diff = (frame - bg.view(1, 1, c)).abs().mean(dim=-1)
                diff_u8 = (diff.numpy() * 255).clip(0, 255).astype(np.uint8)
                _, mask_u8 = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                char_mask = torch.from_numpy(mask_u8) > 127

            ys, xs = torch.where(char_mask)
            if len(ys) == 0:
                infos.append(None)
                continue

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            cropped = frame[y1:y2, x1:x2, :]
            ch, cw = cropped.shape[:2]

            gray = (cropped.mean(dim=-1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                                   minSize=(max(1, cw // 8), max(1, ch // 8)))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2
                method = "anime_cascade"
            else:
                face_cx = cw // 2
                face_cy = ch // 2
                method = "fallback"

            infos.append((cropped, ch, cw, face_cx, face_cy, method))

        # ── Scale computed per-image from character crop ──
        scale = None  # computed per image below

        # ── Pass 2: scale and position each character ──
        box_top = (output_size - target) // 2
        box_bottom = box_top + target
        results = []

        for i in range(n):
            if infos[i] is None:
                results.append(torch.ones(output_size, output_size, c))
                continue

            cropped, ch, cw, face_cx, face_cy, method = infos[i]
            img_scale = min(target / ch, target / cw)
            sh = max(1, int(ch * img_scale))
            sw = max(1, int(cw * img_scale))

            scaled = torch.nn.functional.interpolate(
                cropped.unsqueeze(0).permute(0, 3, 1, 2),
                size=(sh, sw), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).squeeze(0)

            face_cy_scaled = int(face_cy * img_scale)
            face_cx_scaled = int(face_cx * img_scale)

            oy = (output_size // 2) - face_cy_scaled
            oy = max(box_top, min(box_bottom - sh, oy))
            oy = max(0, min(oy, output_size - sh))

            ox = (output_size // 2) - face_cx_scaled
            ox = max(0, min(ox, output_size - sw))

            canvas = torch.ones(output_size, output_size, c)
            canvas[oy:oy + sh, ox:ox + sw, :] = scaled

            print(f"[LoopStrip] [{i}] {method}: {cw}x{ch} → {sw}x{sh} "
                  f"@ ({ox},{oy}) scale={img_scale:.3f} face=({face_cx},{face_cy})")

            results.append(canvas)

        return (torch.stack(results),)


# ─── Split Grid ─────────────────────────────────────────────────────────────


class LoopStripSplitGrid:
    """Split a video grid (e.g. 2x2) into separate video outputs."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "cols": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",) * 4
    RETURN_NAMES = ("cell_1", "cell_2", "cell_3", "cell_4")
    FUNCTION = "execute"
    CATEGORY = "LoopStrip"

    def execute(self, image, cols=2, rows=2):
        n, h, w, c = image.shape
        cell_h = h // rows
        cell_w = w // cols

        cells = []
        for r in range(rows):
            for col_i in range(cols):
                y1 = r * cell_h
                x1 = col_i * cell_w
                cells.append(image[:, y1:y1 + cell_h, x1:x1 + cell_w, :])

        while len(cells) < 4:
            cells.append(torch.zeros(n, 1, 1, c))

        return tuple(cells)


# ─── Registration ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoopStripLoadVideo": LoopStripLoadVideo,
    "LoopStripFindCycle": LoopStripFindCycle,
    "LoopStripCenterSubject": LoopStripCenterSubject,
    "LoopStripCenterCharacter": LoopStripCenterCharacter,
    "LoopStripAssemble": LoopStripAssemble,
    "LoopStripSplitGrid": LoopStripSplitGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopStripLoadVideo": "Loop Strip — Load Video",
    "LoopStripFindCycle": "Loop Strip — Find Best Cycle",
    "LoopStripCenterSubject": "Loop Strip — Center Subject",
    "LoopStripCenterCharacter": "Loop Strip — Center Character",
    "LoopStripAssemble": "Loop Strip — Assemble Sprite Strip",
    "LoopStripSplitGrid": "Loop Strip — Split Grid",
}
