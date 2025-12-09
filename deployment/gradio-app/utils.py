import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io


def calculate_metrics(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    min_val, max_val = float(gray.min()), float(gray.max())
    michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0

    return {
        "Mean Intensity": float(gray.mean()),
        "Std Deviation": float(gray.std()),
        "Entropy": float(entropy),
        "Contrast": float(michelson_contrast)
    }


def create_histogram(img_array, title="Histogram", *, color="gray", metric_type="intensity"):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    fig, ax = plt.subplots(figsize=(6, 4))

    if metric_type == "intensity":
        ax.hist(gray.ravel(), bins=256, range=[0, 256], color=color, alpha=0.7)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
    elif metric_type == "gradient":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        ax.hist(gradient_mag.ravel(), bins=100, color=color, alpha=0.7)
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Frequency')
    elif metric_type == "local_entropy":
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
        local_ent = entropy(gray, disk(5))
        ax.hist(local_ent.ravel(), bins=50, color=color, alpha=0.7)
        ax.set_xlabel('Local Entropy')
        ax.set_ylabel('Frequency')
    elif metric_type == "local_contrast":
        kernel_size = 5
        mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray.astype(np.float64)**2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(sqr_mean - mean**2, 0))
        ax.hist(local_std.ravel(), bins=100, color=color, alpha=0.7)
        ax.set_xlabel('Local Contrast (Std Dev)')
        ax.set_ylabel('Frequency')
    elif metric_type == "cumulative":
        hist, bins = np.histogram(gray.ravel(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        ax.plot(bins[:-1], cdf_normalized, color=color, linewidth=2)
        ax.fill_between(bins[:-1], cdf_normalized, alpha=0.3, color=color)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Probability')

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)


def generate_histograms(gray_img, clahe_img, colored_img, metric_type="intensity"):
    metric_labels = {
        "intensity": "Intensity",
        "gradient": "Gradient",
        "local_entropy": "Local Entropy",
        "local_contrast": "Local Contrast",
        "cumulative": "Cumulative"
    }
    label = metric_labels.get(metric_type, "Intensity")

    return (
        create_histogram(gray_img, f"Original - {label}", color="#6c757d", metric_type=metric_type),
        create_histogram(clahe_img, f"CLAHE - {label}", color="#d9480f", metric_type=metric_type),
        create_histogram(colored_img, f"Colormap - {label}", color="#136f63", metric_type=metric_type)
    )


def format_metrics_dataframe(metrics, clahe_info):
    config_df = pd.DataFrame({
        'Parameter': ['CLAHE Clip Limit', 'CLAHE Tile Grid Size', 'Colormap'],
        'Value': [clahe_info['clipLimit'], str(clahe_info['tileGridSize']), clahe_info['colormap']]
    })

    orig = metrics.get('Original', {})
    colormap = metrics.get('CLAHE + Colormap', {})

    metrics_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Entropy', 'Contrast'],
        'Original': [
            f"{orig.get('Mean Intensity', 0):.2f}",
            f"{orig.get('Std Deviation', 0):.2f}",
            f"{orig.get('Entropy', 0):.2f}",
            f"{orig.get('Contrast', 0):.3f}"
        ],
        'CLAHE + Colormap': [
            f"{colormap.get('Mean Intensity', 0):.2f}",
            f"{colormap.get('Std Deviation', 0):.2f}",
            f"{colormap.get('Entropy', 0):.2f}",
            f"{colormap.get('Contrast', 0):.3f}"
        ]
    })

    return config_df, metrics_df


def _parse_prediction(pred_str):
    pred = str(pred_str).replace('tensor(', '').replace(')', '').lower()
    is_normal = pred == '0' or 'normal' in pred
    return pred, is_normal


def format_prediction_dataframe(result_dict, stage_1_key, stage_2_key):
    stage1 = result_dict.get(stage_1_key, {})
    stage2 = result_dict.get(stage_2_key, {})

    if 'Prediction' not in stage1:
        return pd.DataFrame([['No prediction', '-']], columns=['Result', 'Confidence'])

    pred1, is_normal = _parse_prediction(stage1['Prediction'])

    rows = []

    if is_normal:
        stage1_result = 'Normal'
        stage1_confidence = float(stage1.get('Normal', 0)) * 100
    else:
        stage1_result = 'Pneumonia'
        stage1_confidence = float(stage1.get('Pneumonia', 0)) * 100

    rows.append(['Stage 1', stage1_result, f'{stage1_confidence:.1f}%'])

    if not is_normal and 'Prediction' in stage2:
        pred2 = str(stage2['Prediction']).lower()
        if 'bacterial' in pred2 or pred2 == '0':
            stage2_result = 'Bacterial'
            stage2_confidence = float(stage2.get('Bacterial', 0)) * 100
        else:
            stage2_result = 'Viral'
            stage2_confidence = float(stage2.get('Viral', 0)) * 100
        rows.append(['Stage 2', stage2_result, f'{stage2_confidence:.1f}%'])

    return pd.DataFrame(rows, columns=['Stage', 'Result', 'Confidence'])


def load_image_if_exists(path):
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def format_batch_summary_markdown(batch_size, clahe_config):
    clip = clahe_config.get('clipLimit', 2.0)
    tile = clahe_config.get('tileGridSize', '(8, 8)')
    colormap = clahe_config.get('colormap', 'HOT')
    return f"""| Setting | Value |
|---------|-------|
| **Images** | {batch_size} |
| **CLAHE Clip** | {clip} |
| **Tile Size** | {tile} |
| **Colormap** | {colormap} |"""


def get_diagnosis_info(result_dict, stage_1_key, stage_2_key):
    stage1 = result_dict.get(stage_1_key, {})
    stage2 = result_dict.get(stage_2_key, {})

    pred1, is_normal = _parse_prediction(stage1.get('Prediction', ''))

    if is_normal:
        return {
            'diagnosis': 'Normal',
            'confidence': float(stage1.get('Normal', 0)) * 100,
            'color': '#28a745',
            'color_rgb': (40, 167, 69),
            'icon': '✓'
        }
    elif 'pneumonia' in pred1 or pred1 == '1':
        pred2 = str(stage2.get('Prediction', '')).lower()
        if 'bacterial' in pred2 or pred2 == '0':
            return {
                'diagnosis': 'Pneumonia - Bacterial',
                'confidence': float(stage2.get('Bacterial', 0)) * 100,
                'color': '#fd7e14',
                'color_rgb': (253, 126, 20),
                'icon': '⚠'
            }
        else:
            return {
                'diagnosis': 'Pneumonia - Viral',
                'confidence': float(stage2.get('Viral', 0)) * 100,
                'color': '#dc3545',
                'color_rgb': (220, 53, 69),
                'icon': '⚠'
            }
    return {
        'diagnosis': 'Unknown',
        'confidence': 0,
        'color': '#6c757d',
        'color_rgb': (108, 117, 125),
        'icon': '?'
    }


def create_composite_image(img, diagnosis_info):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    scale_factor = width / 400.0
    font_size = max(12, int(16 * scale_factor))
    header_height = max(25, int(30 * scale_factor))

    new_height = height + header_height
    composite = Image.new('RGB', (width, new_height), (255, 255, 255))
    composite.paste(img, (0, 0))

    draw = ImageDraw.Draw(composite)
    header_top = height
    color_rgb = diagnosis_info.get('color_rgb', (108, 117, 125))
    draw.rectangle([0, header_top, width, new_height], fill=color_rgb)

    icon = diagnosis_info.get('icon', '?')
    diagnosis = diagnosis_info.get('diagnosis', 'Unknown')
    confidence = diagnosis_info.get('confidence', 0)
    text = f"{icon} {diagnosis} - {confidence:.1f}%"

    font = None
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue

    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (width - text_width) // 2
    text_y = header_top + (header_height - text_height) // 2

    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return composite
