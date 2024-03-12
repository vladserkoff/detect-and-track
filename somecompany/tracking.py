"""Vehicle tracking and counting utilities."""

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Set, Tuple

from ultralytics.engine.results import Boxes

from somecompany.logger import logging

LOG = logging.getLogger(__name__)


class Direction(Enum):
    """Vehicle track direction."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"
    LEFT = "left"
    RIGHT = "right"


class VehicleTracker:
    """Vehicle detections tracker.

    Keeps track(😱) of the vehicle tracks(😱😱) in a video stream. A track (😱😱😱)is registered when a vehicle is detected
    at least `min_detections` times anr removed when it has not been detected for `max_track_age` frames. Additionally estimates
    the direction of the vehicle based on the track history (currently only in- and out-bound trajectories are supported).
    """

    tracks: Dict[int, List[Tuple[float, float]]]  # track_id -> list of centroids
    frames_since_last_hit: Dict[int, int]  # track_id -> frames since last seen
    incoming_ids: Set[int]  # track_ids of incoming vehicles
    outgoing_ids: Set[int]  # track_ids of outgoing vehicles

    def __init__(self, max_track_age: int = 30, min_detections: int = 5) -> None:
        """Initialize the tracker.
        Args:
            max_track_age (int): Maximum age of a track (in frames).
            min_detections (int): Minimum number of detections to create a track (in frames).
        Attributes:
            tracks (Dict[int, List[Tuple[float, float]]]): The tracks.
            frames_since_last_hit (Dict[int, int]): Number of frames since the last detection for each track.
            incoming_ids (Set[int]): Track IDs of incoming vehicles.
            outgoing_ids (Set[int]): Track IDs of outgoing vehicles.
        """
        self.max_track_age = max_track_age
        self.min_detections = min_detections
        self.tracks = defaultdict(lambda: [])
        self.frames_since_last_hit = defaultdict(lambda: 0)
        self.incoming_ids = set()
        self.outgoing_ids = set()

    @property
    def num_incoming(self) -> int:
        """Number of incoming vehicles."""
        return len(self.incoming_ids)

    @property
    def num_outgoing(self) -> int:
        """Number of outgoing vehicles."""
        return len(self.outgoing_ids)

    def update(self, detections: Boxes) -> None:
        """Update the tracker with a new frame.

        Args:
            detections (ultralytics.engine.results.Boxes): Detections (with track ids) from the current frame.

        """
        if detections.id is None:
            LOG.debug("No detections in frame, skipping update")
            return
        # Get track IDs and the detection centers
        track_ids = detections.id.int().cpu().tolist()
        boxes = detections.xywh.cpu()
        # boxes are in xywh format (center x, center y + width and height)
        centroids = boxes[:, :2].tolist()

        # Update the tracks
        for centroid, track_id in zip(centroids, track_ids):
            self.tracks[track_id].append(tuple(centroid))
            self.frames_since_last_hit[track_id] = 0
            LOG.debug(f"Track {track_id} updated with centroid {centroid}")

        # Remove old tracks and update the direction counts
        for track_id, track in list(self.tracks.items()):
            if self.frames_since_last_hit[track_id] > self.max_track_age:
                del self.tracks[track_id]
                LOG.debug(f"Track {track_id} removed due to age")
            else:
                self.frames_since_last_hit[track_id] += 1
                LOG.debug(f"Track {track_id} age increased to {self.frames_since_last_hit[track_id]}")
            if len(track) >= self.min_detections:
                if estimate_track_direction(track) == Direction.INCOMING:
                    self.incoming_ids.add(track_id)
                    self.outgoing_ids.discard(track_id)  # Remove from outgoing if it was there
                else:
                    self.outgoing_ids.add(track_id)
                    self.incoming_ids.discard(track_id)  # Remove from incoming if it was there


def estimate_track_direction(track: List[Tuple[float, float]]) -> Direction:
    """Estimate the direction of a track based on the y-coordinate of the first and last detections.

    Args:
        track (List[Tuple[float, float]]): The track to estimate the direction of.

    Returns:
        str: The estimated direction of the track.
    """
    if track[0][1] < track[-1][1]:
        return Direction.INCOMING
    else:
        return Direction.OUTGOING
