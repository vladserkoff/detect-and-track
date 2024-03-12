import unittest

import numpy as np

from somecompany.viz import plot_trajectories


class TestPlotTrajectories(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_empty_trajectories(self):
        trajectories = []
        output_image = plot_trajectories(self.image, trajectories)
        np.testing.assert_array_equal(output_image, self.image)

    def test_single_trajectory(self):
        trajectories = [np.array([[10, 10], [20, 20]])]
        output_image = plot_trajectories(self.image, trajectories)
        self.assertNotEqual(np.sum(output_image), 0)

    def test_multiple_trajectories(self):
        trajectories = [np.array([[10, 10], [20, 20]]), np.array([[30, 30], [40, 40]])]
        output_image = plot_trajectories(self.image, trajectories)
        self.assertNotEqual(np.sum(output_image), 0)

    def test_labels(self):
        trajectories = [np.array([[10, 10], [20, 20]])]
        labels = [0]
        output_image = plot_trajectories(self.image, trajectories, labels)
        self.assertNotEqual(np.sum(output_image), 0)


if __name__ == "__main__":
    unittest.main()
