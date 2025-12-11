from fastai.vision.all import *
from pathlib import Path

def load_pneumonia_learners(model_path: str,
                            stage1_name: str = "set2_pneumonia_detector_final_310.pkl",
                            stage2_name: str = "set2_stage2_bacterial_viral_detector_final_310.pkl"):
    """
    Helper to load stage 1 and stage 2 learners from a directory.
    """
    model_path = Path(model_path)
    learn_stage1 = load_learner(model_path/stage1_name)
    learn_stage2 = load_learner(model_path/stage2_name)
    return learn_stage1, learn_stage2


def run_pipeline_check(img_orig, learn_stage1, learn_stage2,
                       thresh_stage1: float = 0.80,
                       thresh_stage2: float = 0.65):
    """
    Two-stage pneumonia pipeline:

    Stage 1: Normal vs Pneumonia
    Stage 2: (if Pneumonia) Bacterial vs Viral

    Returns:
        final_label (str),
        final_conf (float),
        probs_stage1 (Tensor),
        probs_stage2 (Tensor or None),
        proc_img_stage1 (Tensor),
        proc_img_stage2 (Tensor or None)
    """
    img = PILImage.create(img_orig)

    #Predict normal vs pneumonia
    _, _, probs_1 = learn_stage1.predict(img)
    pneumonia_idx = learn_stage1.dls.vocab.o2i['pneumonia']
    prob_pneumonia = probs_1[pneumonia_idx].item()
    conf_1 = prob_pneumonia

    dl1 = learn_stage1.dls.test_dl([img])
    xb1 = dl1.one_batch()[0]           

    # decode image
    proc_img_stage1 = learn_stage1.dls.after_batch.decode((xb1,))[0][0].cpu()

    if prob_pneumonia < thresh_stage1:
        pred_1 = 'normal'
        return pred_1.capitalize(), conf_1, probs_1, None, proc_img_stage1, None

    print("Stage 2 Vocab:", learn_stage2.dls.vocab)  # e.g., might output ['bacteria', 'virus']
    print("Stage 2 o2i:", learn_stage2.dls.vocab.o2i)  # Shows dict like {'bacteria': 0, 'virus': 1}
    #Predict viral vs bacterial
    _, _, probs_2 = learn_stage2.predict(img)
    viral_idx = learn_stage2.dls.vocab.o2i['viral']
    prob_viral = probs_2[viral_idx].item()
    conf_2 = max(prob_viral, 1 - prob_viral)

    if prob_viral >= thresh_stage2:
        pred_2 = 'viral'
    else:
        pred_2 = 'bacterial'

    dl2 = learn_stage2.dls.test_dl([img])
    xb2 = dl2.one_batch()[0]
    # decode image
    proc_img_stage2 = learn_stage2.dls.after_batch.decode((xb2,))[0][0].cpu()

    return pred_2.capitalize(), conf_2, probs_1, probs_2, proc_img_stage1, proc_img_stage2
