"""Checks that pipeline_model.py agrees with its equations. Run: python -m unittest -v"""
import random
import unittest
import torch
from attention import dense_attention
from pipeline_model import (Hardware, closed_form, decode_attention_staged, decode_items,
                            simulate, stage_cycles)


class FunctionalModelTests(unittest.TestCase):
    def test_staged_decode_matches_dense(self):
        # One query against a cache of 37 keys; tiles that divide, don't divide, and exceed L.
        torch.manual_seed(17)
        heads, length, dim = 3, 37, 8
        k, v = (torch.randn(heads, length, dim, dtype=torch.float64) for _ in range(2))
        for scale in (1.0, 50.0):   # 50x makes large scores, exercising the running max
            q = scale * torch.randn(heads, dim, dtype=torch.float64)
            # dense_attention uses (B, N, H, D); a decode query has N = 1.
            expected = dense_attention(q[None, None], k.transpose(0, 1)[None],
                                       v.transpose(0, 1)[None])[0, 0]
            for tile in (1, 5, 16, 64):
                torch.testing.assert_close(decode_attention_staged(q, k, v, tile), expected)


class TimingModelTests(unittest.TestCase):
    def test_stage_cycles_are_work_over_rate(self):
        hw = Hardware(bytes_per_cycle=64, macs_per_cycle=64, exps_per_cycle=16)
        # LOAD 2*32*64*2/64, SCORE 32*64/64, SOFTMAX 32/16, ACCUMULATE 33*64/64
        self.assertEqual(stage_cycles(32, 64, 2, hw), (128, 32, 2, 33))

    def test_partial_last_tile(self):
        items = decode_items(context=70, heads=2, head_dim=64, tile=32, bytes_per_element=2)
        self.assertEqual(len(items), 2 * 3)                  # heads * ceil(70 / 32)
        self.assertEqual(items[2], stage_cycles(6, 64, 2))   # 70 = 32 + 32 + 6

    def test_simulator_matches_closed_form(self):
        rng = random.Random(17)
        for _ in range(200):
            tau = tuple(rng.randint(1, 20) for _ in range(rng.randint(1, 6)))
            n = rng.randint(1, 40)
            expected = closed_form(tau, n)
            self.assertEqual(simulate([tau] * n, pipelined=False)['cycles'], expected['sequential'])
            self.assertEqual(simulate([tau] * n)['cycles'], expected['pipelined'])

    def test_utilizations_sum_to_speedup(self):
        items = decode_items(context=1000, heads=4, head_dim=64, tile=48, bytes_per_element=2)
        sequential, pipelined = simulate(items, pipelined=False), simulate(items)
        self.assertAlmostEqual(sum(pipelined['utilization']),
                               sequential['cycles'] / pipelined['cycles'])
        self.assertAlmostEqual(sum(sequential['utilization']), 1.0)

    def test_bounds_hold_for_unequal_items(self):
        rng = random.Random(3)
        items = [tuple(rng.randint(1, 30) for _ in range(4)) for _ in range(60)]
        pipelined = simulate(items)['cycles']
        # Never faster than the busiest stage's total work or any single item's path,
        # never slower than the non-pipelined baseline.
        self.assertGreaterEqual(pipelined, max(sum(t[k] for t in items) for k in range(4)))
        self.assertGreaterEqual(pipelined, max(sum(t) for t in items))
        self.assertLessEqual(pipelined, simulate(items, pipelined=False)['cycles'])

    def test_balanced_pipeline_approaches_stage_count(self):
        self.assertAlmostEqual(closed_form((10, 10, 10, 10), 10_000)['speedup'], 4, places=2)


if __name__ == '__main__':
    unittest.main()
