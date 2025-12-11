## About This Model
This application demonstrates a **two-stage hierarchical classification pipeline** for pneumonia detection from chest X-rays.

### Two-Stage Classification Pipeline
| Stage | Details |
| --- | --- |
| **Stage 1: Pneumonia Detection** | Binary classification (Normal vs Pneumonia)<br>Detects presence of pneumonia in chest X-rays. Uses ResNet-50 with transfer learning |
| **Stage 2: Pneumonia Type Classification** | Multi-class classification (Bacterial vs Viral)<br>Runs only if Stage 1 predicts pneumonia<br>Distinguishes bacterial vs viral pneumonia |

| End-to-end two-stage pipeline |
| --- |
| ![Pipeline Diagram](gradio_api/file=images/pipeline_diagram.png) |

### Preprocessing Pipeline
All images undergo a multi-stage preprocessing pipeline:

| Step | Description |
| --- | --- |
| **Grayscale Conversion** | Original RGB → Grayscale |
| **CLAHE Application** | Contrast Limited Adaptive Histogram Equalization |
| **Hot Colormap** | Applied for enhanced visualization |

| 1. Grayscale conversion | 2. CLAHE-enhanced grayscale (CLAHE) | 3. CLAHE + Hot colormap |
| --- | --- | --- |
| ![](gradio_api/file=examples/preprocessed/stages/covid_01_1_grayscale.jpg) | ![](gradio_api/file=examples/preprocessed/stages/covid_01_2_clahe.jpg) | ![](gradio_api/file=examples/preprocessed/stages/covid_01_3_colored.jpg) |

**Why CLAHE?**
- Enhances local contrast without over-amplifying noise
- Highlights subtle pathological features

**Why the Hot Colormap?**
- Converts single-channel CLAHE output into a  heatmap so elevated intensities stand out

### Model Architecture
| Item | Details |
| --- | --- |
| **Base Model** | ResNet-50 (pretrained on ImageNet) |
| **Framework** | FastAI / PyTorch |
| **Training Data** | Kaggle Chest X-Ray Images (Pneumonia) |
| **Total Images** | 5,863 chest X-rays |
| **Classes (Stage 1)** | Normal, Pneumonia
| **Sub Classes (Stage 2)**| Bacterial Pneumonia, Viral Pneumonia |
| **Split** | 80/20 stratified train/validation |


### Important Disclaimer
**Research demonstration only. NOT for medical diagnosis.**

- Educational and research purposes
- Model trained on limited dataset
- COVID-19 samples not in training data
- Not for clinical decision-making

### Technical Stack
| Category | Components |
| --- | --- |
| **Built With** | FastAI, PyTorch, Gradio, OpenCV<br>ResNet-50 pretrained backbone |
| **Preprocessing** | PIL (image loading)<br>OpenCV (CLAHE)<br>scikit-image (entropy)<br>matplotlib (histograms) |

### References & Links
- [GitHub Repository](https://github.com/JLichwa80/image-classification)
- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Author: [@JLichwa80](https://github.com/JLichwa80)

**Citations**:
1. Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with FastAI and PyTorch*. O'Reilly Media. [link](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
2. Waheed, S., et al. (2022). Pre-processing methods in chest X-ray image classification. *Frontiers in Medicine*, 9.  [link](https://www.frontiersin.org/articles/10.3389/fmed.2022.843211)
3. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*, 770-778 [link](https://arxiv.org/abs/1512.03385)
4. Lim, H.-W., et al. (2017). Automatic x-ray contrast enhancement. *Medical Physics*, 44(5). [link](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.12123)

**License**: MIT License
