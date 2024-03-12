import unittest

import numpy as np

from somecompany.trajectory import compute_distance_matrix


class TestComputeDistanceMatrix(unittest.TestCase):
    def test_empty_input(self):
        trajectories = np.empty((0, 2))
        expected_output = np.empty((0, 0))
        np.testing.assert_array_equal(compute_distance_matrix(trajectories), expected_output)

    def test_single_trajectory(self):
        trajectories = np.array([[(0, 0), (1, 1)]])
        expected_output = np.zeros((1, 1))
        np.testing.assert_array_equal(compute_distance_matrix(trajectories), expected_output)

    def test_multiple_trajectories(self):
        trajectories = np.array([[(0, 0), (1, 1)], [(0, 1), (1, 0)]])
        expected_output = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_almost_equal(compute_distance_matrix(trajectories), expected_output, decimal=3)


if __name__ == "__main__":
    unittest.main()
