"""Run with python -m unittest -v. Requires PyTorch; no GPU needed."""
import unittest
import torch
from attention import dense_attention, online_attention
from models import SelfAttentionLayer, TinyTransformer


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_uniform_scores_average_values(self):
        q = torch.zeros(1, 3, 1, 2, dtype=torch.float64)
        k = torch.ones_like(q)
        v = torch.tensor([[[[1., 2.]], [[3., 4.]], [[8., 9.]]]], dtype=q.dtype)
        for implementation in (dense_attention, online_attention):
            torch.testing.assert_close(implementation(q, k, v),
                                       v.mean(dim=1, keepdim=True).expand_as(v))
            expected = v.cumsum(dim=1) / torch.arange(1, 4).view(1, 3, 1, 1)
            torch.testing.assert_close(implementation(q, k, v, causal=True), expected)

    def test_online_matches_outputs_and_gradients(self):
        # Odd length and non-dividing tile sizes exercise partial blocks.
        for causal in (False, True):
            for tile in (1, 3, 16):
                inputs = [torch.randn(2, 7, 2, 4, dtype=torch.float64,
                                      requires_grad=True) for _ in range(3)]
                dense = dense_attention(*inputs, causal=causal)
                online = online_attention(*inputs, causal=causal, block_size=tile)
                torch.testing.assert_close(dense, online, atol=1e-10, rtol=1e-10)
                upstream = torch.randn_like(dense)
                gd = torch.autograd.grad(dense, inputs, upstream)
                go = torch.autograd.grad(online, inputs, upstream)
                for a, b in zip(gd, go):
                    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    def test_causal_future_invariance(self):
        q, k, v = [torch.randn(1, 7, 2, 4) for _ in range(3)]
        for implementation in (dense_attention, online_attention):
            before = implementation(q, k, v, causal=True)
            k2, v2 = k.clone(), v.clone()
            k2[:, 4:] += 100
            v2[:, 4:] -= 100
            after = implementation(q, k2, v2, causal=True)
            torch.testing.assert_close(before[:, :4], after[:, :4])

    def test_reference_gradcheck(self):
        inputs = tuple(torch.randn(1, 3, 1, 2, dtype=torch.float64,
                                   requires_grad=True) for _ in range(3))
        self.assertTrue(torch.autograd.gradcheck(
            lambda *xs: dense_attention(*xs, causal=True), inputs))

    def test_models_match_outputs_and_gradients(self):
        for cls in (SelfAttentionLayer, TinyTransformer):
            for causal in (False, True):
                dense = cls(heads=2, dim=4, method='dense').double()
                online = cls(heads=2, dim=4, method='online', block_size=3).double()
                online.load_state_dict(dense.state_dict())
                x = torch.randn(1, 7, 8, dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_(True)
                a, b = dense(x, causal), online(y, causal)
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                upstream = torch.randn_like(a)
                a.backward(upstream)
                b.backward(upstream)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
                for pd, po in zip(dense.parameters(), online.parameters()):
                    torch.testing.assert_close(pd.grad, po.grad, atol=1e-9, rtol=1e-9)

    def test_transformer_sgd_updates_match(self):
        dense = TinyTransformer(heads=2, dim=4, method='dense').double()
        online = TinyTransformer(heads=2, dim=4, method='online', block_size=3).double()
        online.load_state_dict(dense.state_dict())
        x = torch.randn(1, 7, 8, dtype=torch.float64)
        target = torch.randn_like(x)
        for model in (dense, online):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(x, True), target).backward()
                optimizer.step()
        for pd, po in zip(dense.parameters(), online.parameters()):
            torch.testing.assert_close(pd, po, atol=1e-9, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
