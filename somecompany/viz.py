"""Visualization tools for the system."""

from typing import Any, Optional, Sequence

import cv2
import matplotlib as mpl
import numpy as np

from somecompany.tracking import VehicleTracker

COLORMAP = mpl.color_sequences["tab20"]


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), transparency: float = 0.5
) -> np.ndarray:
    """Overlay buinary mask on an image.
    Args:
        image (np.ndarray): Image to overlay the mask on.
        mask (np.ndarray): Binary mask to overlay.
        color (tuple, optional): Color of the mask. Defaults to blue.
        transparency (float, optional): Transparency of the mask.
    """
    overlay = image.copy()
    overlay[mask != 0] = color
    image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
    return image


def plot_vehicle_tracks(image: np.ndarray, tracker: VehicleTracker) -> np.ndarray:
    """Plot the vehicle tracks on the image.

    Args:
        image (np.ndarray): Image to plot the tracks on.
        tracker (VehicleTracker): Tracker to get the tracks from.

    Returns:
        np.ndarray: Image with the tracks plotted on.
    """
    # Draw the tracking lines
    tracks = []
    track_ids = []
    for track_id, track in tracker.tracks.items():
        if len(track) < 2:
            continue
        tracks.append(np.array(track))
        track_ids.append(track_id)
    image = plot_trajectories(image, trajectories=tracks, labels=track_ids)
    return image


def plot_trajectories(
    image: np.ndarray, trajectories: Sequence[np.ndarray], labels: Optional[Sequence[int]] = None, copy: bool = False
) -> np.ndarray:
    """Plot the trajectories on the image.

    Args:
        image (np.ndarray): Image to plot the trajectories on.
        trajectories (np.ndarray): Trajectories to plot, represented by points in [x, y] format.
        labels (Optional[np.ndarray]): Labels of the trajectories, e.g. the track ids.

    Returns:
        np.ndarray: Image with the trajectories plotted on.
    """
    if copy:
        image = image.copy()
    for i, trajectory in enumerate(trajectories):
        points = trajectory.astype(np.int32).reshape((-1, 1, 2))
        if labels is not None:
            label = labels[i]
            color = COLORMAP[label % len(COLORMAP)]
            # convert color to unit8 BGR as required by cv2
            color = np.array(color[::-1]) * 255
        else:
            label = None
            color = (0, 0, 255)
        cv2.polylines(image, [points], False, color, thickness=1)
        if label is not None:
            cv2.putText(image, str(label), tuple(points[-1, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
    return image
