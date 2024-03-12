import unittest

from somecompany.tracking import Direction, estimate_track_direction


class TestEstimateTrackDirection(unittest.TestCase):
    def test_incoming_direction(self):
        track = [(0, 0), (1, 1)]
        self.assertEqual(estimate_track_direction(track), Direction.INCOMING)

    def test_outgoing_direction(self):
        track = [(0, 1), (1, 0)]
        self.assertEqual(estimate_track_direction(track), Direction.OUTGOING)

    def test_same_y_coordinates(self):
        track = [(0, 0), (1, 0)]
        self.assertEqual(estimate_track_direction(track), Direction.OUTGOING)


if __name__ == "__main__":
    unittest.main()
