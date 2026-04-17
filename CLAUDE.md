# Loop Strip — Notes for Claude

ComfyUI custom node package for character animation: walk cycle detection, face-based centering, sprite assembly, grid splitting.

## Architecture

All nodes live in `nodes.py`. Each class:
- Defines `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION = "execute"`, `CATEGORY = "LoopStrip"`
- Is registered in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` at the bottom of the file
- `__init__.py` re-exports those mappings

ComfyUI passes IMAGE tensors as `[N, H, W, C]` float32 in 0..1 range.

## Key design decisions

- **Face detection: lbpcascade_animeface** (bundled XML). Heuristic approaches (fixed ratios, width profile / neck detection, skin color, DWPose, eye detection, symmetry) were tried 15+ times and ALL failed on chibi characters. The cascade is the only reliable approach. Don't reintroduce heuristics.
- **Center Character scaling**: per-image crop-based — `scale = min(target / ch, target / cw)`. Source-image-based scaling was tried and broke when characters had different padding in their source frames.
- **Center Character positioning**: face centered on canvas, but clamped so character stays inside the target box (`fill_percent * output_size`).
- **Find Best Cycle grid mode**: when `grid_cols`/`grid_rows` > 1, the cycle is detected ONCE on the full video, then the same frame indices are applied to each cell (keeps multi-character grids in sync).

## Outputs

- Find Best Cycle: 5 outputs — `loop_frames` (always works on full input) + `grid_1..grid_4` (only when grid > 1x1).
- Split Grid: 4 outputs (`cell_1..cell_4`).
- Center Character: single IMAGE output.

## Publishing updates

See `~/.claude/projects/-Users-serhiiyashyn/memory/reference_comfyui_publish.md`.

Quick version:
1. Bump `version` in `pyproject.toml`
2. `git add . && git commit -m "..." && git push`
3. `/Users/serhiiyashyn/Library/Python/3.9/bin/comfy --skip-prompt --here node publish --token <PAT>`

## Gotchas

- `__pycache__/` is gitignored but ComfyUI doesn't hot-reload — user must restart ComfyUI to pick up code changes.
- The `lbpcascade_animeface.xml` file MUST be present in the package directory; loaded via `os.path.join(os.path.dirname(__file__), 'lbpcascade_animeface.xml')`.
- When face detection fails (back/side views), the fallback uses the **head region** (top 40% of mask) largest-connected-component centroid for horizontal centering — this ignores weapons/arms extending sideways. Vertical is fixed at `ch * 0.43`, which matches what the cascade finds on front-views of chibi characters with large hair/crowns (bbox top sits well above the real eyes). Don't try using the head blob's Y: head blobs in a top-slice stretch to the slice boundary (neck is continuous with head), so blob-height × ratio collapses to a fraction of the cutoff instead of tracking the real face.
