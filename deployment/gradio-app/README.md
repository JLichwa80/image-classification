---
title: Pneumonia Detection - Two-Stage Classification
emoji: 🫁
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Pneumonia Detection - Two-Stage Classification

Interactive demo for pneumonia detection from chest X-ray images using a two-stage hierarchical classification pipeline.

## Model Overview

This application uses **ResNet-50 transfer learning** with **CLAHE preprocessing** to classify chest X-rays in two stages:

![Complete Pipeline Diagram](../../res/architecture.png)

### Stage 1: Pneumonia Detection
- **Task**: Binary classification
- **Classes**: Normal vs Pneumonia
- **Architecture**: ResNet-50 (pretrained on ImageNet)

### Stage 2: Pneumonia Type Classification
- **Task**: Multi-class classification (only runs if pneumonia detected)
- **Classes**: Bacterial vs Viral Pneumonia
- **Architecture**: ResNet-50 (pretrained on ImageNet)

## Preprocessing

All images are preprocessed using:
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
  - clipLimit: 2.0
  - tileGridSize: (8, 8)
- **Hot Colormap** for enhanced visualization

This preprocessing significantly improves contrast and highlights important features in chest X-rays.

## Features

- **Single Image Analysis**: Upload and analyze one X-ray at a time
- **Batch Processing**: Upload and analyze up to 10 X-rays simultaneously
- **Visual Feedback**: See the preprocessed image with CLAHE + Hot colormap applied
- **Detailed Results**: Get probability scores for each classification stage

## How to Use

### Single Image Mode
1. Click on the "Single Image" tab
2. Upload a chest X-ray image (JPEG, PNG, etc.)
3. Click "Analyze Image"
4. View the preprocessed image and prediction results

### Batch Mode
1. Click on the "Multiple Images (Batch)" tab
2. Upload multiple chest X-ray images (up to 10)
3. Click "Analyze Batch"
4. View all preprocessed images in the gallery and batch results

## Performance

The model achieves high accuracy on test data:
- **Stage 1**: Robust pneumonia detection with balanced precision and recall
- **Stage 2**: Accurate classification of bacterial vs viral pneumonia

See the [GitHub repository](https://github.com/JLichwa80/image-classification) for detailed performance metrics.

## Training Data

- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,863 chest X-rays
- **Classes**: Normal, Bacterial Pneumonia, Viral Pneumonia
- **Split**: 80/20 stratified train/validation split

## Important Disclaimer

**This is a research demonstration only. Not intended for medical diagnosis.**

- This model is for educational and research purposes
- Always consult qualified medical professionals for diagnosis
- The model was trained on a limited dataset and may not generalize to all cases
- COVID-19 samples are shown for demonstration but were not part of the training data

## Technical Details

**Built With:**
- [FastAI](https://www.fast.ai/) - Deep learning library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gradio](https://gradio.app/) - Web interface
- [OpenCV](https://opencv.org/) - Image preprocessing
- ResNet-50 (pretrained on ImageNet)

**Preprocessing Pipeline:**
1. Convert to grayscale
2. Apply CLAHE for contrast enhancement
3. Apply Hot colormap for visualization
4. Feed to ResNet-50 model

## Links

- **GitHub Repository**: https://github.com/JLichwa80/image-classification
- **Dataset**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Author**: [@JLichwa80](https://github.com/JLichwa80)

## License

This project is licensed under the MIT License.

## Citation

If you use this model or code in your research, please cite:

```bibtex
@software{pneumonia_detection_2025,
  author = {JLichwa80},
  title = {Pneumonia Detection - Two-Stage Classification},
  year = {2025},
  url = {https://github.com/JLichwa80/image-classification}
}
```

## References

1. Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with FastAI and PyTorch*. O'Reilly Media.

2. Waheed, S., Ghosh, S., & Gadekallu, T. R. (2022). Pre-processing methods in chest X-ray image classification. *Frontiers in Medicine*, 9, 898289.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

4. Lim, H.-W., et al. (2017). Automatic x‐ray image contrast enhancement based on parameter optimization using entropy. *Medical Physics*, 44(5), 2212–2226.
