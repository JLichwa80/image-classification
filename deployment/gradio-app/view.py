import gradio as gr
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from utils import format_batch_summary_markdown, get_diagnosis_info, create_composite_image, generate_histograms

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def create_single_image_tab(default_stage_images, default_hist_images, default_sample_image, default_prediction_df, default_config_df, default_metrics_df, default_composite_img):
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Select Image")
            single_input = gr.Image(
                type="pil",
                label="Upload X-Ray",
                value=default_sample_image,
                height=200
            )
            single_button = gr.Button("Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=3, min_width=350):
            gr.Markdown("### Prediction Results")
            with gr.Row():
                with gr.Column(scale=1):
                    stage3_img = gr.Image(
                        label="",
                        type="pil",
                        height=250,
                        value=default_composite_img,
                        container=True,
                        show_label=False,
                        elem_classes="diagnosis-card"
                    )
                with gr.Column(scale=1):
                    single_result = gr.Dataframe(
                        label="",
                        interactive=False,
                        value=default_prediction_df,
                        wrap=True,
                        elem_classes="prediction-table",
                        show_label=False
                    )

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Image Processing Configuration")
            config_display = gr.Dataframe(
                label="",
                interactive=False,
                value=default_config_df,
                wrap=True,
                elem_classes="compact-table",
                show_label=False
            )
        with gr.Column(scale=2):
            gr.Markdown("#### Image Quality Metrics")
            metrics_display = gr.Dataframe(
                label="",
                interactive=False,
                value=default_metrics_df,
                wrap=True,
                elem_classes="compact-table",
                show_label=False
            )

    gr.Markdown("---")
    gr.Markdown("### Preprocessing Pipeline")
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("**Step 1: Grayscale Conversion**")
            stage1_img = gr.Image(label="", type="pil", value=default_stage_images[0], height=300, show_label=False)
        with gr.Column():
            gr.Markdown("**Step 2: CLAHE Enhancement**")
            stage2_img = gr.Image(label="", type="pil", value=default_stage_images[1], height=300, show_label=False)
        with gr.Column():
            gr.Markdown("**Step 3: CLAHE + Colormap**")
            stage3_step_img = gr.Image(label="", type="pil", value=default_stage_images[2], height=300, show_label=False)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Distribution Analysis")
        with gr.Column(scale=1):
            histogram_metric = gr.Dropdown(
                choices=[
                    ("Pixel Intensity", "intensity"),
                    ("Gradient Magnitude", "gradient"),
                    ("Local Entropy", "local_entropy"),
                    ("Local Contrast", "local_contrast"),
                    ("Cumulative Distribution", "cumulative")
                ],
                value="intensity",
                label="Histogram Metric",
                interactive=True
            )
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("**Original**")
            hist1_img = gr.Image(label="", type="pil", value=default_hist_images[0], height=280, show_label=False)
        with gr.Column():
            gr.Markdown("**After CLAHE**")
            hist2_img = gr.Image(label="", type="pil", value=default_hist_images[1], height=280, show_label=False)
        with gr.Column():
            gr.Markdown("**After Colormap**")
            hist3_img = gr.Image(label="", type="pil", value=default_hist_images[2], height=280, show_label=False)

    return {
        'input': single_input,
        'button': single_button,
        'result': single_result,
        'config': config_display,
        'metrics': metrics_display,
        'stage3_step': stage3_step_img,
        'stage1': stage1_img,
        'stage2': stage2_img,
        'stage3': stage3_img,
        'hist1': hist1_img,
        'hist2': hist2_img,
        'hist3': hist3_img,
        'histogram_metric': histogram_metric
    }


def create_batch_analysis_tab(default_batch_gallery, default_batch_summary_md):
    gr.Markdown("## Batch Upload & Analysis\nUpload multiple chest X-ray images (up to 10)")

    with gr.Row():
        with gr.Column(scale=1):
            batch_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload X-Ray Images (up to 10)"
            )
            batch_button = gr.Button("Analyze Batch", variant="primary", interactive=False, size="lg")

        with gr.Column(scale=1):
            gr.Markdown("#### Configuration")
            batch_summary = gr.Markdown(value=default_batch_summary_md or "*Upload images to see summary*")

    gr.Markdown("---")
    gr.Markdown("### Results")

    with gr.Row():
        with gr.Column(scale=3):
            batch_gallery_colored = gr.Gallery(
                label="Processed X-Rays with Predictions",
                columns=4,
                rows=2,
                height="auto",
                value=default_batch_gallery,
                object_fit="contain",
                show_label=False,
                elem_classes="batch-gallery",
                allow_preview=True
            )

        with gr.Column(scale=1):
            gr.Markdown("#### Histograms")
            batch_histogram_metric = gr.Dropdown(
                choices=[
                    ("Pixel Intensity", "intensity"),
                    ("Gradient Magnitude", "gradient"),
                    ("Local Entropy", "local_entropy"),
                    ("Local Contrast", "local_contrast"),
                    ("Cumulative Distribution", "cumulative")
                ],
                value="intensity",
                label="Histogram Metric",
                interactive=True
            )
            gr.Markdown("*Click an image to view histograms*")
            batch_hist_original = gr.Image(label="Original", type="pil", height=150, show_label=True)
            batch_hist_clahe = gr.Image(label="CLAHE", type="pil", height=150, show_label=True)
            batch_hist_colormap = gr.Image(label="Colormap", type="pil", height=150, show_label=True)

    batch_stage_images = gr.State(value=[])
    batch_selected_index = gr.State(value=None)

    return {
        'input': batch_input,
        'button': batch_button,
        'stage_images': batch_stage_images,
        'selected_index': batch_selected_index,
        'summary': batch_summary,
        'gallery_colored': batch_gallery_colored,
        'hist_original': batch_hist_original,
        'hist_clahe': batch_hist_clahe,
        'hist_colormap': batch_hist_colormap,
        'histogram_metric': batch_histogram_metric
    }


def load_default_images(examples_dir):
    stage_paths = [
        examples_dir / "preprocessed" / "stages" / "covid_01_1_grayscale.jpg",
        examples_dir / "preprocessed" / "stages" / "covid_01_2_clahe.jpg",
        examples_dir / "preprocessed" / "stages" / "covid_01_3_colored.jpg",
    ]
    default_stage_images = [Image.open(p).convert("RGB") for p in stage_paths]

    hist_paths = [
        examples_dir / "preprocessed" / "histograms" / "covid_01_hist_gray.png",
        examples_dir / "preprocessed" / "histograms" / "covid_01_hist_clahe.png",
        examples_dir / "preprocessed" / "histograms" / "covid_01_hist_colormap.png",
    ]
    default_hist_images = [Image.open(p).convert("RGB") for p in hist_paths]

    batch_colormap_paths = [
        examples_dir / "preprocessed" / "stages" / "covid_01_3_colored.jpg",
        examples_dir / "preprocessed" / "stages" / "covid_02_3_colored.jpg",
        examples_dir / "preprocessed" / "stages" / "covid_03_3_colored.jpg",
    ]
    default_batch_colormap_images = [Image.open(p).convert("RGB") for p in batch_colormap_paths]

    STAGE_1_KEY = "Stage 1: Pneumonia Detection"
    STAGE_2_KEY = "Stage 2: Pneumonia Type"

    default_results_path = examples_dir / "default_results" / "covid_01_results.json"
    with open(default_results_path, 'r') as f:
        data = json.load(f)

    default_prediction_df = pd.DataFrame(data['prediction_df'])
    default_config_df = pd.DataFrame(data['config_df'])
    default_metrics_df = pd.DataFrame(data['metrics_df'])

    prediction = data.get('prediction', {})
    diagnosis_info = get_diagnosis_info(prediction, STAGE_1_KEY, STAGE_2_KEY)
    default_composite_img = create_composite_image(default_stage_images[2], diagnosis_info)

    default_sample_image = Image.open(examples_dir / "covid_01.jpg").convert("RGB")

    all_results_path = examples_dir / "default_results" / "all_results.json"
    with open(all_results_path, 'r') as f:
        all_data = json.load(f)

    batch_predictions = []
    for img_name, img_data in all_data.items():
        pred = img_data.get('prediction', {})
        batch_predictions.append({
            STAGE_1_KEY: pred.get(STAGE_1_KEY, {}),
            STAGE_2_KEY: pred.get(STAGE_2_KEY, {})
        })

    clahe_config = {
        'clipLimit': CLAHE_CLIP_LIMIT,
        'tileGridSize': str(CLAHE_TILE_GRID_SIZE),
        'colormap': 'HOT'
    }
    default_batch_summary_md = format_batch_summary_markdown(len(all_data), clahe_config)

    default_batch_gallery = []
    for img, result in zip(default_batch_colormap_images, batch_predictions):
        info = get_diagnosis_info(result, STAGE_1_KEY, STAGE_2_KEY)
        composite_img = create_composite_image(img, info)
        default_batch_gallery.append(composite_img)

    return {
        'stage_images': default_stage_images,
        'hist_images': default_hist_images,
        'batch_gallery': default_batch_gallery,
        'prediction_df': default_prediction_df,
        'config_df': default_config_df,
        'metrics_df': default_metrics_df,
        'sample_image': default_sample_image,
        'composite_img': default_composite_img,
        'batch_summary_md': default_batch_summary_md
    }


def create_app_ui(content_dir, examples_dir, counter_message_fn, handle_single_fn, handle_batch_fn):
    def md(filename: str) -> str:
        return (content_dir / filename).read_text(encoding="utf-8")

    defaults = load_default_images(examples_dir)

    with gr.Blocks(title="Pneumonia Detection from Chest X-Rays") as demo:
        gr.HTML("""
        <style>
        footer {display: none !important;}
        .compact-table table, .prediction-table table {
            font-size: 1.1rem;
            border-collapse: collapse;
            width: 100%;
        }
        .compact-table th, .prediction-table th {
            background-color: #f8f9fa;
            padding: 12px 16px;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            font-size: 1.1rem;
        }
        .compact-table td, .prediction-table td {
            padding: 10px 16px;
            border-bottom: 1px solid #e9ecef;
            font-size: 1.05rem;
        }
        .prediction-table td:last-child {
            font-weight: bold;
        }
        .diagnosis-card {
            border-radius: 10px 10px 0 0 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        .gradio-container {
            max-width: 100% !important;
            padding: 0.5rem !important;
        }
        .block {
            margin-bottom: 0.5rem !important;
        }
        .batch-gallery .gallery-item {
            border: 3px solid #dee2e6 !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        .batch-gallery img {
            object-fit: contain !important;
        }
        .diagnosis-card img {
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        </style>
        <script>
        function highlightPredictions() {
            document.querySelectorAll('.prediction-table td').forEach(cell => {
                const text = cell.textContent.trim();
                if (text === 'Normal') {
                    cell.style.backgroundColor = '#d4edda';
                    cell.style.color = '#155724';
                } else if (text === 'Pneumonia') {
                    cell.style.backgroundColor = '#f8d7da';
                    cell.style.color = '#721c24';
                }
            });
        }
        setTimeout(highlightPredictions, 1000);
        const observer = new MutationObserver(highlightPredictions);
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """)
        gr.Markdown(md("hero.md"))

        counter_display = gr.Textbox(label="Session Statistics", value=counter_message_fn(), interactive=False)

        with gr.Tabs():
            with gr.Tab("Single Image Analysis"):
                single = create_single_image_tab(
                    defaults['stage_images'],
                    defaults['hist_images'],
                    defaults['sample_image'],
                    defaults['prediction_df'],
                    defaults['config_df'],
                    defaults['metrics_df'],
                    defaults['composite_img']
                )

            with gr.Tab("Batch Analysis"):
                batch = create_batch_analysis_tab(
                    defaults['batch_gallery'],
                    defaults['batch_summary_md']
                )

            with gr.Tab("About"):
                gr.Markdown(md("about.md"))

        single_analyzed = gr.State(value=False)
        stage_images_state = gr.State(value=None)

        def analyze_and_mark_done(input_image):
            results = handle_single_fn(input_image)
            stage_imgs = (results[0], results[1], results[2]) if results[0] is not None else None
            return results + (True, gr.update(interactive=False), stage_imgs)

        single['button'].click(
            fn=analyze_and_mark_done,
            inputs=single['input'],
            outputs=[single['stage1'], single['stage2'], single['stage3_step'],
                     single['hist1'], single['hist2'], single['hist3'],
                     single['result'], single['config'], single['metrics'],
                     single['stage3'], counter_display, single_analyzed, single['button'], stage_images_state]
        )

        def regenerate_histograms(metric_type, stage_imgs):
            if stage_imgs is None:
                return None, None, None
            gray_img, clahe_img, colored_img = stage_imgs
            return generate_histograms(
                np.array(gray_img), np.array(clahe_img), np.array(colored_img), metric_type
            )

        single['histogram_metric'].change(
            fn=regenerate_histograms,
            inputs=[single['histogram_metric'], stage_images_state],
            outputs=[single['hist1'], single['hist2'], single['hist3']]
        )

        single['input'].change(
            fn=lambda img: (gr.update(interactive=img is not None), False),
            inputs=single['input'],
            outputs=[single['button'], single_analyzed]
        )

        batch['button'].click(
            fn=handle_batch_fn,
            inputs=batch['input'],
            outputs=[batch['gallery_colored'], batch['stage_images'], batch['summary'], counter_display]
        )

        batch['input'].change(
            fn=lambda files: gr.update(interactive=bool(files)),
            inputs=batch['input'],
            outputs=batch['button']
        )

        def show_batch_histograms(stage_images, metric_type, evt: gr.SelectData):
            if stage_images and evt.index < len(stage_images):
                gray_img, clahe_img, colored_img = stage_images[evt.index]
                return generate_histograms(
                    np.array(gray_img), np.array(clahe_img), np.array(colored_img), metric_type
                ) + (evt.index,)
            return None, None, None, None

        batch['gallery_colored'].select(
            fn=show_batch_histograms,
            inputs=[batch['stage_images'], batch['histogram_metric']],
            outputs=[batch['hist_original'], batch['hist_clahe'], batch['hist_colormap'], batch['selected_index']]
        )

        def regenerate_batch_histograms(metric_type, stage_images, selected_index):
            if stage_images is None or selected_index is None or selected_index >= len(stage_images):
                return None, None, None
            gray_img, clahe_img, colored_img = stage_images[selected_index]
            return generate_histograms(
                np.array(gray_img), np.array(clahe_img), np.array(colored_img), metric_type
            )

        batch['histogram_metric'].change(
            fn=regenerate_batch_histograms,
            inputs=[batch['histogram_metric'], batch['stage_images'], batch['selected_index']],
            outputs=[batch['hist_original'], batch['hist_clahe'], batch['hist_colormap']]
        )

    return demo
