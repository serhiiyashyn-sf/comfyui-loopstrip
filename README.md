# ComfyUI Loop Strip

Character animation nodes for ComfyUI: walk cycle detection, face-based centering, and sprite strip assembly.

## Nodes

- **Loop Strip — Load Video** — load a video file into a batch of frames
- **Loop Strip — Find Best Cycle** — detects the best seamless loop cycle in a walking animation; optional grid splitting for multi-character videos
- **Loop Strip — Center Subject** — centers a moving subject across video frames
- **Loop Strip — Center Character** — centers a character image using anime face detection (lbpcascade_animeface)
- **Loop Strip — Assemble Sprite Strip** — concatenates frames into a horizontal or vertical sprite strip
- **Loop Strip — Split Grid** — splits a grid video (e.g. 2x2) into separate videos

## Installation

Via ComfyUI Manager, or manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui-loopstrip
cd comfyui-loopstrip
pip install -r requirements.txt
```

## License

MIT
