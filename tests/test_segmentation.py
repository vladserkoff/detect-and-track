import unittest

import cv2
import numpy as np

from somecompany.segmentation import postprocess_mask, test_intersection


class TestPostprocessMask(unittest.TestCase):
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        processed_mask = postprocess_mask(mask)
        np.testing.assert_array_equal(processed_mask, mask)

    def test_single_blob_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
        processed_mask = postprocess_mask(mask)
        np.testing.assert_array_equal(processed_mask, mask)

    def test_multiple_blobs_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
        cv2.rectangle(mask, (30, 30), (50, 50), 255, -1)
        processed_mask = postprocess_mask(mask)
        expected_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(expected_mask, (30, 30), (50, 50), 255, -1)
        np.testing.assert_array_equal(processed_mask, expected_mask)


class TestIntersection(unittest.TestCase):
    def test_no_intersection(self):
        bboxes_xyxy = np.array([[0, 0, 1, 1]])
        mask = np.zeros((3, 3), dtype=np.uint8)
        self.assertFalse(test_intersection(bboxes_xyxy, mask).any())

    def test_single_intersection(self):
        bboxes_xyxy = np.array([[0, 0, 1, 1]])
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[0, 0] = 255
        self.assertTrue(test_intersection(bboxes_xyxy, mask).any())

    def test_multiple_intersections(self):
        bboxes_xyxy = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[0, 0] = 255
        mask[1, 1] = 255
        self.assertTrue(all(test_intersection(bboxes_xyxy, mask)))


if __name__ == "__main__":
    unittest.main()
