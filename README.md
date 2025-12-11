# Pneumonia Detection from Chest X-rays

> Deep learning classification of chest X-ray images to detect pneumonia and distinguish between bacterial and viral infections.

---

## Project Overview

This project uses transfer learning with ResNet-50 to classify chest X-ray images in a hierarchical two-stage pipeline.

- **Objective:** Detect pneumonia from chest X-rays and classify bacterial vs. viral pneumonia
- **Domain:** Medical Imaging
- **Key Techniques:** Transfer Learning (ResNet-50), CLAHE Preprocessing, Two-Stage Classification, Data Augmentation

---

## Project Structure

```
├── data/                                   # Dataset CSV files and images
├── code/
│   └── Pneumonia.ipynb                     # Main analysis notebook
├── models/                                 # Trained models
├── deployment/
│   └── gradio-app/                         # Lightweight Gradio demo for inference
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## Data

- **Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Description:** 5,863 chest X-ray images categorized as Normal, Bacterial Pneumonia, or Viral Pneumonia
- **Preprocessing:** Two approaches tested - baseline grayscale and CLAHE (Contrast Limited Adaptive Histogram Equalization) with Hot colormap enhancement

---

## Analysis

The project implements a two-stage hierarchical classification pipeline using transfer learning with pretrained ResNet-50:
- **Stage 1:** Binary classification (Normal vs Pneumonia)
- **Stage 2:** Pneumonia subtype classification (Bacterial vs Viral)

![Complete Pipeline Diagram](res/architecture.png)

Two training sets are compared:
- **Set 1:** Baseline grayscale X-rays with moderate augmentation
- **Set 2:** CLAHE-enhanced X-rays with aggressive augmentation

Training uses stratified 80/20 train/validation splits with class-weighted loss functions to handle data imbalance. Run `code/PneumoniaDetector.ipynb` to reproduce the full pipeline including preprocessing, training, and evaluation.

---

## Results

The two-stage pipeline successfully classifies chest X-rays with both preprocessing approaches:
- **Stage 1 (Normal vs Pneumonia):** High accuracy across both training sets with CLAHE preprocessing showing enhanced contrast and entropy
- **Stage 2 (Bacterial vs Viral):** Focal loss function effectively handles class imbalance between bacterial and viral pneumonia cases

Detailed metrics including accuracy, precision, recall, F1-scores, and confusion matrices are available in the results folder.

---

## Setup and Installation

### 1. Clone Repository to Google Drive

Clone or download this repository to your Google Drive.

### 2. Download Dataset

Download the [Kaggle Chest X-Ray Images dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract the images to the `data/` folder within the repository.

### 3. Mount Google Drive and Update Project Path

Open `code/PneumoniaDetector.ipynb` in Google Colab and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update the `PROJECT_PATH` variable to point to your repository location:

```python
PROJECT_PATH = '/content/drive/MyDrive/path/to/your/image-classification'
```

### 4. Run in Google Colab with GPU

This notebook requires GPU for efficient training:

1. Go to **Runtime → Change runtime type**
2. Select **GPU** as the hardware accelerator
3. Run all cells sequentially

---

## Authors

- [@JLichwa80](https://github.com/JLichwa80)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

### Tools & Libraries
- Python (numpy, pandas, torch, fastai, opencv-python, scikit-learn, matplotlib, seaborn)
- FastAI library for transfer learning framework
- PyTorch for deep learning
- Google Colab for GPU computing
- Jupyter Notebook

### Datasets
- Kermany, D., Zhang, K., & Goldbaum, M. (2018). Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. *Mendeley Data*, v2. [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- COVID-19 test images from [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

### Research Papers
1. Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with FastAI and PyTorch*. O'Reilly Media.

2. Waheed, S., Ghosh, S., & Gadekallu, T. R. (2022). Pre-processing methods in chest X-ray image classification. *Frontiers in Medicine*, 9, 898289. https://doi.org/10.3389/fmed.2022.898289

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

4. Panwar, H. et al. (2020). A deep learning and grad-CAM based color visualization approach for fast detection of COVID-19 cases using chest X-ray and CT-Scan images. *Chaos Solitons Fractals*, 140, 110190.

5. Lim, H.-W., et al. (2017). Automatic x‐ray image contrast enhancement based on parameter optimization using entropy. *Medical Physics*, 44(5), 2212–2226. doi:10.1002/acm2.12172
