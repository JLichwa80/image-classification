import cv2
import numpy as np
import pandas as pd
from PIL import Image
import logging

# Import the prediction pipeline (same as Colab)
from pneumonia_detector_pipeline import run_pipeline_check

logger = logging.getLogger(__name__)

from utils import (
    calculate_metrics, generate_histograms, format_metrics_dataframe,
    format_prediction_dataframe, format_batch_summary_markdown,
    get_diagnosis_info, create_composite_image
)



def preprocessing_data(img, clahe_settings):
    pil_img = img.convert("RGB")
    original = np.array(pil_img)

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    clip = clahe_settings.get("clipLimit", 2.0)
    grid = clahe_settings.get("tileGridSize", (8, 8))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(grid))
    clahe_enhanced = clahe.apply(gray)

    cmap_name = clahe_settings.get("colormap", "COLORMAP_HOT")
    cmap_attr = cmap_name if cmap_name.startswith("COLORMAP_") else f"COLORMAP_{cmap_name.upper()}"
    colormap = getattr(cv2, cmap_attr, cv2.COLORMAP_HOT)
    colored = cv2.applyColorMap(clahe_enhanced, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    clahe_rgb = cv2.cvtColor(clahe_enhanced, cv2.COLOR_GRAY2RGB)

    snapshots = [
        Image.fromarray(gray_rgb),
        Image.fromarray(clahe_rgb),
        Image.fromarray(colored_rgb)
    ]

    metrics = {
        "Original": calculate_metrics(gray),
        "CLAHE": calculate_metrics(clahe_enhanced),
        "CLAHE + Colormap": calculate_metrics(colored_rgb)
    }

    histograms = generate_histograms(gray, clahe_enhanced, colored_rgb)

    return snapshots, metrics, histograms


def predict_single(input_image, stage1_model, stage2_model, clahe_settings, classification_icons,
                   stage_1_key, stage_2_key, counter_message_fn, increment_counter_fn):
    if input_image is None:
        return (None, None, None, None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                None, counter_message_fn())

    snapshots, metrics, histograms, result = _run_model_pipeline(
        input_image, stage1_model, stage2_model, clahe_settings, stage_1_key, stage_2_key
    )

    gray_img, clahe_img, colored_img = snapshots
    hist_original, hist_clahe, hist_colormap = histograms

    config_df, metrics_df = format_metrics_dataframe(metrics, clahe_settings)
    result_df = format_prediction_dataframe(result, stage_1_key, stage_2_key)

    diagnosis_info = get_diagnosis_info(result, stage_1_key, stage_2_key)
    composite_img = create_composite_image(colored_img, diagnosis_info)

    increment_counter_fn(1)

    return (gray_img, clahe_img, colored_img, hist_original, hist_clahe, hist_colormap,
            result_df, config_df, metrics_df, composite_img, counter_message_fn())


def predict_batch(image_files, stage1_model, stage2_model, clahe_settings,
                  stage_1_key, stage_2_key, counter_message_fn, increment_counter_fn):
    if not image_files:
        return [], [], "*Upload images to see summary*", counter_message_fn()

    image_files = image_files[:10]

    images_colored = []
    all_stage_images = []

    for img_file in image_files:
        with Image.open(img_file.name) as img:
            pil_img = img.convert("RGB")

        snapshots, metrics, histograms, result = _run_model_pipeline(
            pil_img, stage1_model, stage2_model, clahe_settings, stage_1_key, stage_2_key
        )

        gray_img, clahe_img, colored_img = snapshots
        diagnosis_info = get_diagnosis_info(result, stage_1_key, stage_2_key)
        composite_img = create_composite_image(colored_img, diagnosis_info)
        images_colored.append(composite_img)
        all_stage_images.append((gray_img, clahe_img, colored_img))

    increment_counter_fn(len(image_files))

    summary_md = format_batch_summary_markdown(len(image_files), clahe_settings)

    return images_colored, all_stage_images, summary_md, counter_message_fn()


def _run_model_pipeline(pil_img, stage1_model, stage2_model, clahe_settings, stage_1_key, stage_2_key):
    snapshots, metrics, histograms = preprocessing_data(pil_img, clahe_settings)

    dl1 = stage1_model.dls.test_dl([pil_img])
    _, preds1, _, dec1 = stage1_model.get_preds(dl=dl1, with_input=True, with_decoded=True)
    probs1 = preds1[0]
    pred1 = dec1[0]
    vocab1 = stage1_model.dls.vocab
    logger.info(f"[STAGE1] Vocab: {vocab1}")
    logger.info(f"[STAGE1] Raw probs: {probs1}")
    logger.info(f"[STAGE1] Result: {pred1} ({vocab1[0]}={float(probs1[0]):.2%}, {vocab1[1]}={float(probs1[1]):.2%})")

    result = {
        stage_1_key: {
            "Prediction": str(pred1),
            "Normal": f"{float(probs1[0]):.4f}",
            "Pneumonia": f"{float(probs1[1]):.4f}"
        }
    }

    # pred1 could be tensor index (0/1) or class name ('normal'/'pneumonia')
    vocab1 = stage1_model.dls.vocab
    # Handle tensor, int, or string
    pred1_val = pred1.item() if hasattr(pred1, 'item') else pred1
    if isinstance(pred1_val, int) or str(pred1_val).isdigit():
        pred1_label = vocab1[int(pred1_val)].lower()
    else:
        pred1_label = str(pred1_val).lower()
    logger.info(f"[STAGE1] pred1 raw: {pred1}, converted: {pred1_val}, label: '{pred1_label}'")
    if pred1_label == 'pneumonia':
        dl2 = stage2_model.dls.test_dl([pil_img])
        _, preds2, _, dec2 = stage2_model.get_preds(dl=dl2, with_input=True, with_decoded=True)
        probs2 = preds2[0]
        pred2 = dec2[0]
        vocab2 = stage2_model.dls.vocab
        logger.info(f"[STAGE2] Vocab: {vocab2}")
        logger.info(f"[STAGE2] Raw probs: {probs2}")
        logger.info(f"[STAGE2] Result: {pred2} ({vocab2[0]}={float(probs2[0]):.2%}, {vocab2[1]}={float(probs2[1]):.2%})")
        result[stage_2_key] = {
            "Prediction": str(pred2),
            "Bacterial": f"{float(probs2[0]):.4f}",
            "Viral": f"{float(probs2[1]):.4f}"
        }
    else:
        logger.info(f"[STAGE2] Skipped - no pneumonia detected")
        result[stage_2_key] = {"Note": "Stage 2 skipped - no pneumonia detected"}

    return snapshots, metrics, histograms, result
``