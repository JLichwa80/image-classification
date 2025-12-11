import cv2
import numpy as np
import pandas as pd
from PIL import Image

import logging
from custom_transforms import EnsureGrayscale, CLAHETransform, ColormapTransform, FastFocalLoss

# Import the working prediction pipeline
from pneumonia_detector_pipeline import run_pipeline_check


logger = logging.getLogger(__name__)

from utils import (
    calculate_metrics, generate_histograms, format_metrics_dataframe,
    format_prediction_dataframe, format_batch_summary_markdown,
    get_diagnosis_info, create_composite_image
)


# =============================================================================
# VISUALIZATION/PREPROCESSING FUNCTIONS (Histograms, metrics, image stages)
# =============================================================================


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


# =============================================================================
# UI FUNCTIONS (Combine prediction + visualization for Gradio)
# =============================================================================

def _convert_prediction_to_result_dict(prediction, confidence, probs_1, probs_2,
                                        stage1_model, stage2_model, stage_1_key, stage_2_key):
    """
    Convert run_pipeline_check output to the result dict format used by UI.
    """
    # Get probabilities from tensors
    pneumonia_idx = stage1_model.dls.vocab.o2i['pneumonia']
    normal_idx = stage1_model.dls.vocab.o2i['normal']
    prob_pneumonia = probs_1[pneumonia_idx].item()
    prob_normal = probs_1[normal_idx].item()

    result = {
        stage_1_key: {
            "Prediction": "Pneumonia" if prob_pneumonia >= 0.5 else "Normal",
            "Normal": f"{prob_normal:.4f}",
            "Pneumonia": f"{prob_pneumonia:.4f}"
        }
    }

    if probs_2 is not None:
        viral_idx = stage2_model.dls.vocab.o2i['viral']
        bacterial_idx = stage2_model.dls.vocab.o2i['bacterial']
        prob_viral = probs_2[viral_idx].item()
        prob_bacterial = probs_2[bacterial_idx].item()

        result[stage_2_key] = {
            "Prediction": prediction,
            "Bacterial": f"{prob_bacterial:.4f}",
            "Viral": f"{prob_viral:.4f}"
        }
    else:
        result[stage_2_key] = {"Note": "Stage 2 skipped - no pneumonia detected"}

    return result


def predict_single(input_image, stage1_model, stage2_model, clahe_settings, classification_icons,
                   stage_1_key, stage_2_key, counter_message_fn, increment_counter_fn):
    """Single image prediction with visualization."""
    if input_image is None:
        return (None, None, None, None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                None, counter_message_fn())

    # Step 1: Get prediction using run_pipeline_check (returns 6 values)
    prediction, confidence, probs_1, probs_2, proc_img_1, proc_img_2 = run_pipeline_check(
        input_image, stage1_model, stage2_model
    )

    # Step 2: Get visualization (histograms, metrics, image stages)
    snapshots, metrics, histograms = preprocessing_data(input_image, clahe_settings)

    # Step 3: Convert prediction to result dict for UI
    result = _convert_prediction_to_result_dict(
        prediction, confidence, probs_1, probs_2,
        stage1_model, stage2_model, stage_1_key, stage_2_key
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
    """Batch image prediction with visualization."""
    if not image_files:
        return [], [], "*Upload images to see summary*", counter_message_fn()

    image_files = image_files[:10]

    images_colored = []
    all_stage_images = []

    for img_file in image_files:
        with Image.open(img_file.name) as img:
            pil_img = img.convert("RGB")

        # Step 1: Get prediction (returns 6 values)
        prediction, confidence, probs_1, probs_2, proc_img_1, proc_img_2 = run_pipeline_check(
            pil_img, stage1_model, stage2_model
        )

        # Step 2: Get visualization
        snapshots, metrics, histograms = preprocessing_data(pil_img, clahe_settings)

        # Step 3: Convert to result dict
        result = _convert_prediction_to_result_dict(
            prediction, confidence, probs_1, probs_2,
            stage1_model, stage2_model, stage_1_key, stage_2_key
        )

        gray_img, clahe_img, colored_img = snapshots
        diagnosis_info = get_diagnosis_info(result, stage_1_key, stage_2_key)
        composite_img = create_composite_image(colored_img, diagnosis_info)
        images_colored.append(composite_img)
        all_stage_images.append((gray_img, clahe_img, colored_img))

    increment_counter_fn(len(image_files))

    summary_md = format_batch_summary_markdown(len(image_files), clahe_settings)

    return images_colored, all_stage_images, summary_md, counter_message_fn()
