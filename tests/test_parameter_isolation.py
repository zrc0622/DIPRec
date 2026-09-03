import unittest


class ParameterIsolationTest(unittest.TestCase):
    def test_independent_parameters_and_effective_routing(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from diprec.modeling import InterestParameterRouter

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.lm_head = torch.nn.Linear(4, 16, bias=False)

            def get_input_embeddings(self):
                return self.embedding

            def get_output_embeddings(self):
                return self.lm_head

        model = TinyLM()
        router = InterestParameterRouter(model, [12, 13])
        router.assert_parameter_isolation([1, 2, 3])
        self.assertFalse({id(p) for p in router.adapter.parameters()} & {id(p) for p in model.embedding.parameters()})
        output = model.embedding(torch.tensor([[1, 12]]))
        self.assertTrue(torch.equal(output[0, 1], router.adapter.embedding.weight[0]))
        router.close()

    def test_output_route_survives_buffer_sync_between_forwards(self):
        """Model the in-place buffer broadcast DDP performs before each forward."""

        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from diprec.modeling import InterestParameterRouter

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.lm_head = torch.nn.Linear(4, 16, bias=False)

            def get_input_embeddings(self):
                return self.embedding

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, input_ids):
                return self.lm_head(self.embedding(input_ids))

        model = TinyLM()
        router = InterestParameterRouter(model, [12, 13])
        first = model(torch.tensor([[1, 12]]))
        with torch.no_grad():
            router.adapter.global_ids.copy_(router.adapter.global_ids)
        second = model(torch.tensor([[2, 13]]))
        (first.sum() + second.sum()).backward()
        self.assertIsNotNone(router.adapter.output_head.weight.grad)
        router.close()

    def test_joint_forward_routes_plan_and_sid_gradients(self):
        """One autoregressive trajectory trains plan output and SID conditioning."""

        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"tiny Qwen runtime is unavailable: {exc}")

        from diprec.modeling import InterestParameterRouter
        from diprec.sft import _causal_stage_losses

        torch.manual_seed(7)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=32,
                use_cache=False,
            )
        )
        # IDs 24--26 represent the independent interest vocabulary; 5--7 are
        # ordinary SID/EOS vocabulary rows in the base language-model head.
        router = InterestParameterRouter(model, [24, 25, 26])
        input_ids = torch.tensor([[1, 24, 25, 26, 4, 5, 6, 7]])
        labels = torch.tensor([[-100, 24, 25, 26, 4, 5, 6, 7]])
        stage_ids = torch.tensor([[-1, 0, 0, 0, 0, 1, 1, 1]])

        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        plan_loss, sid_loss, plan_tokens, sid_tokens = _causal_stage_losses(
            output.logits, labels, stage_ids
        )
        self.assertEqual((plan_tokens, sid_tokens), (4, 3))

        plan_head_grad = torch.autograd.grad(
            plan_loss,
            router.adapter.output_head.weight,
            retain_graph=True,
        )[0]
        # SID loss must flow through the generated plan prefix's independent
        # input embeddings, which is the key distinction of the joint format.
        sid_conditioning_grad = torch.autograd.grad(
            sid_loss,
            router.adapter.embedding.weight,
            retain_graph=True,
        )[0]
        (0.5 * plan_loss + 0.5 * sid_loss).backward()

        self.assertGreater(plan_head_grad.abs().sum().item(), 0.0)
        self.assertGreater(sid_conditioning_grad.abs().sum().item(), 0.0)
        self.assertGreater(model.lm_head.weight.grad[[5, 6, 7]].abs().sum().item(), 0.0)
        router.close()


if __name__ == "__main__":
    unittest.main()
