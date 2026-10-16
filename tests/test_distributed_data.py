import unittest

from dml_cluster.distributed_data import (
    compute_weighted_slices,
    score_batch_multipliers,
)


class WeightedSliceTests(unittest.TestCase):
    def test_slices_cover_every_sample_without_overlap(self) -> None:
        slices = compute_weighted_slices(
            {"leader": 1.0, "worker-a": 2.0, "worker-b": 3.0},
            total_samples=60,
        )

        self.assertEqual(slices["leader"], (0, 10))
        self.assertEqual(slices["worker-a"], (10, 30))
        self.assertEqual(slices["worker-b"], (30, 60))

    def test_each_participant_receives_a_sample_when_possible(self) -> None:
        slices = compute_weighted_slices(
            {"leader": 100.0, "worker-a": 1.0, "worker-b": 1.0},
            total_samples=3,
        )

        counts = [stop - start for start, stop in slices.values()]
        self.assertEqual(counts, [1, 1, 1])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_weighted_slices({}, total_samples=10)
        with self.assertRaises(ValueError):
            compute_weighted_slices({"leader": 1.0}, total_samples=0)


class BatchMultiplierTests(unittest.TestCase):
    def test_scores_are_scaled_from_the_slowest_participant(self) -> None:
        multipliers = score_batch_multipliers(
            {"leader": 10.0, "worker-a": 25.0, "worker-b": 100.0}
        )

        self.assertEqual(multipliers, {"leader": 1, "worker-a": 2, "worker-b": 8})

    def test_empty_scores_return_empty_mapping(self) -> None:
        self.assertEqual(score_batch_multipliers({}), {})


if __name__ == "__main__":
    unittest.main()
