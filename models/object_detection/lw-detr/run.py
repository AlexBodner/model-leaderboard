import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
import torch
import torchvision.transforms as T
from PIL import Image
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm
import os
from huggingface_hub import list_repo_files, hf_hub_download
from torchvision import transforms
from types import SimpleNamespace


sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing

from configs import CONFIDENCE_THRESHOLD, DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
    run_shell_command
)

# if not Path("D-FINE").is_dir():
#     run_shell_command(
#         ["git", "clone", "https://github.com/Atten4Vis/LW-DETR.git", "./LW-DETR/"]
#     )
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./LW-DETR/")))
from util.utils import ModelEma, BestMetricHolder, clean_state_dict
from util.misc import nested_tensor_from_tensor_list

from models import build_model


REPO_ID = "xbsu/LW-DETR"
SUBDIR = "pretrain_weights"
# 2. List all files in the repo
all_files = list_repo_files(REPO_ID)
weight_files = [f for f in all_files if f.startswith(SUBDIR) and f.endswith(".pth")]

MODEL_DICT = {}

for file_path in weight_files:
    model_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g., "LWDETR_tiny_60e_coco"
    print(f"Downloading: {model_name}")
    
    local_path = hf_hub_download(repo_id=REPO_ID, filename=file_path)
    MODEL_DICT[model_name] = local_path

LICENSE = "Apache-2.0"
HUB_URL = "lyuwenyu/RT-DETR"
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
)
GIT_REPO_URL = "https://github.com/lyuwenyu/RT-DETR"
PAPER_URL = "https://arxiv.org/abs/2304.08069"


TRANSFORMS = T.Compose(
    [T.Resize((RUN_PARAMETERS["imgsz"], RUN_PARAMETERS["imgsz"])), T.ToTensor()]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
            transforms.Resize([640, 640]),
            normalize,
        ])
    image = transform(image)
    return image, orig_image_size
def run_single_model(
    model_id: str,
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    model_values = MODEL_DICT[model_id]

    if skip_if_result_exists and result_json_already_exists(model_id):
        print(f"Skipping {model_id}. Result already exists!")
        return
    if dataset is None:
        dataset = load_detections_dataset(DATASET_DIR)
    local_path = hf_hub_download(repo_id=REPO_ID, filename=file_path)

    cfg = SimpleNamespace(
        hidden_dim=256,
        dec_layers=3,
        num_queries=100,
        encoder='vit_tiny',
        use_ema=False,
        ema_decay=0.9997,
    )
    model, criterion, postprocessors = build_model(cfg)
    checkpoint = torch.load(local_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    model.to(DEVICE)
    model.eval()

    if cfg.use_ema:
        ema_m = ModelEma(model, decay=cfg.ema_decay)
    else:
        ema_m = None
    predictions = []
    targets = []
    print("Evaluating...")
    for img_path, image, target_detections in tqdm(dataset, total=len(dataset)):
        image, orig_image_size = preprocess_image(img_path)
        images = nested_tensor_from_tensor_list([image])
        # forward
        with torch.no_grad():
            outputs = model(images)

        orig_image_sizes = torch.stack([orig_image_size])
        # postprocess
        predictions = postprocessors['bbox'](outputs, orig_image_sizes)

        # visualize
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()


        class_id = labels.detach().cpu().numpy().astype(int)
        xyxy = boxes.detach().cpu().numpy()
        confidence = scores.detach().cpu().numpy()

        detections = sv.Detections(
            xyxy=xyxy[0],
            confidence=confidence[0],
            class_id=class_id[0],
        )

        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    mAP_metric = MeanAveragePrecision()
    f1_metric = F1Score()
    f1_result = f1_metric.update(predictions, targets).compute()
    mAP_result = mAP_metric.update(predictions, targets).compute()

    write_result_json(
        model_id=model_id,
        model_name=model_values["name"],
        model_git_url=GIT_REPO_URL,
        paper_url=PAPER_URL,
        model=model,
        mAP_result=mAP_result,
        f1_score_result=f1_result,
        license_name=LICENSE,
        run_parameters=RUN_PARAMETERS,
    )


def run(
    model_ids: List[str],
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    """
    Run the evaluation for the given models and dataset.

    Arguments:
        model_ids: List of model ids to evaluate. Evaluate all models if None.
        skip_if_result_exists: If True, skip the evaluation if the result json already exists.
        dataset: If provided, use this dataset for evaluation. Otherwise, load the dataset from the default directory.
    """  # noqa: E501 // docs
    if not model_ids:
        model_ids = list(MODEL_DICT.keys())

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")
        process = multiprocessing.Process(
            target=run_single_model, args=(model_id, skip_if_result_exists, dataset)
        )
        process.start()
        process.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_ids",
        nargs="*",
        help="Model ids to evaluate. If not provided, evaluate all models.",
    )
    parser.add_argument(
        "--skip_if_result_exists",
        action="store_true",
        help="If specified, skip the evaluation if the result json already exists.",
    )
    args = parser.parse_args()

    run(args.model_ids, args.skip_if_result_exists)
