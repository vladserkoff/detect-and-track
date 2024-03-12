"""Vehicle tracking pipeline."""

import os
import time
from functools import partial

import cv2
import fire
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from yaml import safe_load

from somecompany.logger import logging
from somecompany.segmentation import RoadPlaneSegmenter, test_intersection
from somecompany.tracking import VehicleTracker
from somecompany.viz import overlay_mask, plot_vehicle_tracks

LOG = logging.getLogger(__name__)


@torch.inference_mode()
def process_video(
    source: str, checkpoint_name: str = "yolov8n.pt", device: str = "cuda", output: str = "screen"
) -> None:
    """Process a video and detect vehicles.
    Args:
        source (str): Path to the video file.
        checkpoint_name (str): Name of the checkpoint to load.
        device (str): Device to run inference on.
        output (str): Output path for the processed video. If "screen" the video will be displayed on screen.
    """

    if not os.path.exists(source):
        ValueError(f"Video path {source} doesn't exist")

    t0_all = time.time()

    with open("config.yml", encoding="utf-8") as f:
        config = safe_load(f)
    config.update({"source": source, "checkpoint_name": checkpoint_name, "device": device, "output": output})
    LOG.info(f"Using config: {config}")

    model = YOLO(checkpoint_name).to(device)
    model.fuse()  # Fuse Conv2d + BatchNorm2d layers
    get_tracks = partial(
        model.track,
        classes=config["coco_relevant_classes"],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=LOG.isEnabledFor(logging.DEBUG),
    )

    road_plane_segmenter = RoadPlaneSegmenter(**config["road_plane_segmenter"])
    tracker = VehicleTracker(**config["tracker"])

    cap = cv2.VideoCapture(source)
    frames_count, fps, width, height = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT),
        cap.get(cv2.CAP_PROP_FPS),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    LOG.info(f"Input video #frames ={frames_count}, fps ={fps}, width ={width}, height={height}")

    frameNumber = 0
    if output == "screen":
        cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
    else:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    bar = tqdm(total=frames_count, desc="Processing frames", unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mask = road_plane_segmenter.update(frame)
        detections: Boxes = get_tracks(frame)[0].boxes

        if detections.id is None:
            LOG.debug("No detections in frame, skipping update")
        else:
            LOG.debug(f"Found {len(detections)} detections in frame")
            keep_detections = test_intersection(detections.xyxy.long().numpy(), mask)
            detections = detections[keep_detections]
            LOG.debug(f"Kept {len(detections)} detections after testing the intersection with the road plane")
            tracker.update(detections)

        frame = overlay_mask(frame, mask)
        frame = plot_vehicle_tracks(frame, tracker)

        current_fps = frameNumber / (time.time() - t0_all)
        cv2.putText(
            frame,
            (
                f"Frame#: {frameNumber}/{int(frames_count)}, incoming: {tracker.num_incoming}, "
                f"outgoing: {tracker.num_outgoing}, FPS: {current_fps:.1f}"
            ),
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (2, 10, 200),
            2,
        )
        if output == "screen":
            cv2.imshow("output", frame)
        else:
            writer.write(frame)
        key = cv2.waitKey(1)
        # Quit when 'q' is pressed
        if key == ord("q") or key == ord("Q") or key == 27:
            break
        elif key == ord("k") or key == ord("K") or key == 32:
            cv2.waitKey(0)
        frameNumber = frameNumber + 1
        bar.update(1)
    cap.release()
    cv2.destroyAllWindows()
    if output != "screen":
        writer.release()
    t1_all = time.time()
    time_taken = t1_all - t0_all
    print(f"Done. process_video took ({time_taken:.3f}s) @ {frameNumber / time_taken:.1f} FPS")


if __name__ == "__main__":
    fire.Fire(process_video)
