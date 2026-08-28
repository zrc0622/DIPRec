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


if __name__ == "__main__":
    unittest.main()
