import unittest

from diprec.runtime import thinking_prompt_ids


class FakeTokenizer:
    def __init__(self, template_ids, decoded_tail):
        self.template_ids = template_ids
        self.decoded_tail = decoded_tail

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return list(self.template_ids)

    def decode(self, ids, skip_special_tokens=False):
        del ids, skip_special_tokens
        return self.decoded_tail

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text != "<think>":
            raise AssertionError(text)
        return [99]


class RuntimeTemplateTest(unittest.TestCase):
    def test_does_not_duplicate_qwen_think_prefix(self):
        tokenizer = FakeTokenizer([1, 2], "assistant\n<think>\n")
        self.assertEqual(thinking_prompt_ids(tokenizer, []), [1, 2])

    def test_adds_think_prefix_for_generic_template(self):
        tokenizer = FakeTokenizer([1, 2], "assistant\n")
        self.assertEqual(thinking_prompt_ids(tokenizer, []), [1, 2, 99])


if __name__ == "__main__":
    unittest.main()
