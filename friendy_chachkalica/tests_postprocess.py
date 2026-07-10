import unittest

import torch

from friendy_chachkalica.postprocess import apply_class_aware_nms


class ClassAwareNmsTests(unittest.TestCase):
    def test_suppresses_same_class_overlap(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.52, 0.5, 0.4, 0.4, 0.8, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.7, 0.0],
        ])

        kept = apply_class_aware_nms(prediction, 0.5)

        self.assertEqual(kept.shape[0], 2)
        self.assertEqual(kept[:, 4].tolist(), [0.9, 0.7])

    def test_keeps_overlapping_different_classes(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.5, 0.5, 0.4, 0.4, 0.8, 1.0],
        ])

        kept = apply_class_aware_nms(prediction, 0.5)

        self.assertEqual(kept.shape[0], 2)
        self.assertEqual(kept[:, 4].tolist(), [0.9, 0.8])

    def test_null_threshold_disables_nms(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.5, 0.5, 0.4, 0.4, 0.8, 0.0],
        ])

        kept = apply_class_aware_nms(prediction, None)

        self.assertTrue(torch.equal(kept, prediction))


if __name__ == "__main__":
    unittest.main()
