#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deepcam-cn/yolov5-face PT weights to ONNX")
    parser.add_argument("--repo", required=True, help="Path to a deepcam-cn/yolov5-face checkout")
    parser.add_argument("--weights", default="pretrain/yolov5s-face.pt")
    parser.add_argument("--output", default="pretrain/yolov5s-face.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    if not (repo / "models" / "experimental.py").is_file():
        raise FileNotFoundError(f"not a yolov5-face checkout: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")

    sys.path.insert(0, str(repo))
    import onnx
    import torch
    import torch.nn as nn
    import models
    from models.experimental import attempt_load
    from utils.activations import Hardswish, SiLU

    # Older YOLOv5-face checkpoints contain model objects, so trusted local
    # weights must be loaded with weights_only disabled on modern PyTorch.
    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_load
    model = attempt_load(str(weights), map_location=torch.device("cpu"))
    detect = model.model[-1]
    if hasattr(detect, "anchor_grid"):
        delattr(detect, "anchor_grid")
    detect.anchor_grid = [torch.zeros(1)] * detect.nl
    detect.export_cat = True
    model.eval()

    for module in model.modules():
        module._non_persistent_buffers_set = set()
        if isinstance(module, models.common.Conv):
            if isinstance(module.act, nn.Hardswish):
                module.act = Hardswish()
            elif isinstance(module.act, nn.SiLU):
                module.act = SiLU()

    image = torch.zeros(1, 3, args.input_size, args.input_size)
    model(image)
    model.fuse()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        image,
        str(output),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        dynamo=False,
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    print(f"exported {output}")


if __name__ == "__main__":
    main()
