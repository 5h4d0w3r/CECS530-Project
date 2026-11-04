"""Checks that cached token generation computes exactly what full recomputation computes."""
import unittest
import torch
from generation import KVCache, TinyLM, generate


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyLM(vocab=50, layers=2, heads=2, dim=4, max_positions=32).double()

    def test_cached_steps_match_full_causal_forward(self):
        # Teacher forcing: feed known tokens (prefill of 5, then one at a time). Every
        # position's logits must equal those of one causal pass over the whole sequence.
        tokens = torch.randint(50, (3, 11))
        with torch.inference_mode():
            full = self.model(tokens)
            cache = self.model.new_cache(3, 11)
            steps = [self.model(tokens[:, :5], cache)]
            steps += [self.model(tokens[:, t:t + 1], cache) for t in range(5, 11)]
        torch.testing.assert_close(torch.cat(steps, dim=1), full)
        self.assertEqual(cache.length, 11)

    def test_greedy_generation_matches_recomputation(self):
        prompt = torch.randint(50, (2, 6))
        cached, seconds = generate(self.model, prompt, 8)
        recomputed, _ = generate(self.model, prompt, 8, use_cache=False)
        self.assertTrue(torch.equal(cached, recomputed))
        self.assertEqual(len(seconds), 8)

    def test_cache_size_formula(self):
        cache = KVCache(layers=2, batch=3, max_len=10, heads=2, dim=4)
        # 2 (K and V) * layers * batch * heads * tokens * head_dim * 4 bytes (FP32)
        self.assertEqual(cache.nbytes(10), 2 * 2 * 3 * 2 * 10 * 4 * 4)
        self.assertEqual(cache.nbytes(10), sum(t.nbytes for t in cache.keys + cache.values))

    def test_cache_limits(self):
        with torch.inference_mode():
            cache = self.model.new_cache(1, 4)
            self.model(torch.randint(50, (1, 4)), cache)
            with self.assertRaises(ValueError):   # no room for a fifth token
                self.model(torch.randint(50, (1, 1)), cache)
            cache = self.model.new_cache(1, 8)
            self.model(torch.randint(50, (1, 3)), cache)
            with self.assertRaises(ValueError):   # several tokens after prefill: unsupported
                self.model(torch.randint(50, (1, 2)), cache)


if __name__ == '__main__':
    unittest.main()
