# Shoplifting Detection Using Dense MViT v2

A high-performance AI-powered shoplifting detection system using Multiscale Vision Transformer (MViT) with FP16 quantization for optimized inference speed.

## Features

- Dense frame-by-frame video analysis using MViT v2 Small
- FP16 quantization for 20-30% inference speedup
- Temporal smoothing with majority voting for robust detection
- GPU-accelerated preprocessing with ImageNet normalization
- High-confidence threshold (80%) to minimize false positives
- Real-time processing capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KrishOberoi/Shoplifting-Detection-Using-Dense-MViTv2.git
cd Shoplifting-Detection-Using-Dense-MViTv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model:
- MViT model checkpoint (`mvit_final_optimized.pth`) - place in the project root or update path in code

## Usage

```python
from src.mvit_dense_detection import run_inference_streaming_fp16_dense

# Load the pre-trained MViT model (automatically loads and converts to FP16)
# The model loading is handled internally in the function

# Run dense frame-by-frame detection with FP16 quantization
run_inference_streaming_fp16_dense(
    source="path/to/your/video.mp4",
    model=None  # Model is loaded internally
)
```

## Project Structure

```
Shoplifting-Detection-Using-Dense-MViTv2/
├── src/
│   ├── __init__.py
│   └── mvit_dense_detection.py   # MViT dense detection with FP16 quantization
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- OpenCV >= 4.8.0
- NumPy >= 1.21.0

## Model Details

- **Architecture**: MViT v2 Small (Multiscale Vision Transformer)
- **Input**: 16-frame video clips at 224x224 resolution
- **Classes**: Binary classification (Normal vs Shoplifting)
- **Precision**: FP16 quantization for 20-30% inference speedup
- **Threshold**: 80% confidence required for shoplifting detection
- **Temporal Coverage**: Dense frame-by-frame analysis with 16-frame sliding window

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
@software{shoplifting_detection_mvit,
  title={Shoplifting Detection Using Dense MViT v2},
  author={Krish Oberoi},
  year={2025},
  url={https://github.com/KrishOberoi/Shoplifting-Detection-Using-Dense-MViTv2}
}
