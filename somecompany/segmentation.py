"""Road plane segmentation."""

from collections import deque
from typing import Deque

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from somecompany.logger import logging

LOG = logging.getLogger(__name__)


class RoadPlaneSegmenter:
    """Segments road plane in frames from a video feed.

    Performs segmentation over several frames to reduce noise and let the cars move away. Takse the mode
    of the segmentation masks over several frames. Assumes that the largest blob in the mask is the road plane and the
    canera is stationary.

    TODO: Make async! Currently blocks the the whole pipeline.
    """

    masks: Deque[torch.tensor]

    def __init__(
        self,
        num_frames: int = 5,
        skip_frames_step_size: int = 0,
        model_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-768-768",
        road_class_name: str = "road",
    ) -> None:
        """Args:
            num_frames (int, optional): Number of frames to use for the segmentation mask.
            skip_frames_step_size (int, optional): Skip every `skip_frames_step_size` frames.
            model_checkpoint (str, optional): Model checkpoint name to use.
            road_class_name (str, optional): Class name of the road plane (usually just road).
        Attributes:
            pipe (transformers.pipelines.Pipeline): Huggingface pipeline for segmentation.
            masks (collections.deque): Sequence of binary masks to compute the mode over.
        """
        super().__init__()
        self.num_frames = num_frames
        self.skip_frames_step_size = skip_frames_step_size
        self.model_checkpoint = model_checkpoint
        self.road_class_name = road_class_name
        self.pipe = pipeline(model=self.model_checkpoint)
        self.masks = deque(maxlen=self.num_frames)  # store masks in a deque to be able to control the size (jic)
        self._mask = None  # precomputed mask
        self._current_frame = 0  # current frame number, used to track skipped frames

    @property
    def segmentation_mask(self) -> np.ndarray:
        """Returns the current segmentation mask."""
        if len(self.masks) == 0:
            raise ValueError("No frames have been processed yet.")
        elif len(self.masks) < self.num_frames:
            # still building up the mask
            mask = torch.mode(torch.stack(tuple(self.masks)), dim=0).values.numpy()
            self._mask = postprocess_mask(mask)
        return self._mask

    @torch.inference_mode()
    def update(self, frame: np.ndarray) -> np.ndarray:
        """Update the segmentation mask with a new frame.
        Args:
            frame (np.ndarray): The frame to update the model with.
        """
        if (len(self.masks) < self.num_frames) and (self._current_frame % self.skip_frames_step_size == 0):
            LOG.info(f"Updating segmentation mask with {len(self.masks) + 1} frames")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = self.pipe(Image.fromarray(image))  # pipeline expects a PIL image
            for pred in preds:
                if pred["label"] == self.road_class_name:
                    # convert array to tensor because computing a mode is much faster on tensors
                    mask = torch.tensor(np.array(pred["mask"]))
                    LOG.debug("Adding road plane mask with %i positive pixels", mask.sum().item())
                    self.masks.append(mask)
                    break
        self._current_frame += 1
        return self.segmentation_mask


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Postprocess the segmentation mask.
    Assumes that the larges blob in the mask is the road plane. Discards all other blobs.

    Args:
        mask (np.ndarray): Raw binary mask with (possibly) many blobs.
    Returns:
        np.ndarray: processed binary mask with extra blobs removed.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    mask = cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask


def test_intersection(bboxes_xyxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tests if the bounding box(es) intersects with the mask.
    Converts bounding boxes into binary mask and returns true if any of the pixels in the mask and the
    bounding boxs are non-zero.

    Args:
        bbox_xywh (np.ndarray): Bounding boxes to test, shape (N, 4).
        mask (np.ndarray): Mask to test against.
    Returns:
        np.ndarray: True if the bounding box intersects with the mask.
    """
    num_boxes = len(bboxes_xyxy)
    bbox_mask = np.zeros((num_boxes, *mask.shape), dtype=mask.dtype)
    for i, bbox in enumerate(bboxes_xyxy):
        bbox_mask[i, bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return np.logical_and(bbox_mask, mask).reshape(num_boxes, -1).any(axis=1)
