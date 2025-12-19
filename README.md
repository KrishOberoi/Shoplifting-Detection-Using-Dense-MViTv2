# Shoplifting Detection System

A professional AI-powered shoplifting detection system using computer vision and deep learning techniques.

## Features

- Video classification with MViT (Multiscale Vision Transformer)
- Full event context extraction for verification
- High accuracy by viewing each frame upto 15 times using a sliding window with a stride 1

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shoplifting-detection.git
cd shoplifting-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
- MViT model checkpoint (`mvit_final_optimized.pth`) - place in the project root or update path in code

## Usage

### Main Detection System
```python
from src.shoplifting_detection import run_detection

# Run detection on a video
run_detection(
    video_path="path/to/your/video.mp4",
    output_path="output.mp4",  # optional
    display=True  # show real-time detection
)
```

### MViT Dense Detection (FP16)
```python
from src.mvit_dense_detection import run_inference_streaming_fp16_dense

# Run dense frame-by-frame detection with FP16 quantization
run_inference_streaming_fp16_dense(
    source="path/to/your/video.mp4",
    model=mvit_model  # Pre-loaded MViT model
)
```

## Project Structure

```
shoplifting-detection/
├── src/
│   ├── __init__.py
│   ├── shoplifting_detection.py  # Main detection code
│   └── mvit_dense_detection.py   # MViT dense detection with FP16 quantization
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- Torchvision

## Model Details

- **Video Classification**: MViT v2 Small for action recognition (quantized to FP16 for optimized performance)
- **Threshold**: 75% confidence required for shoplifting confirmation
- **Precision**: FP16 quantization for 20-30% inference speedup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```
@software{shoplifting_detection,
  title={AI-Powered Shoplifting Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/shoplifting-detection}
}
