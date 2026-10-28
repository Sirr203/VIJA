import unittest

from app.services.prompts import one_step_system, tone_instruction, translate_system


class PromptsToneTests(unittest.TestCase):
    def test_auto_tone(self):
        prompt = translate_system("Vietnamese", "Japanese", tone="auto")
        self.assertIn("casual speech stays casual, polite speech stays polite", prompt)
        self.assertIn("です/ます", prompt)

        # Mặc định khi không truyền tone cũng là auto
        prompt_default = translate_system("Vietnamese", "Japanese")
        self.assertEqual(prompt, prompt_default)

    def test_polite_tone(self):
        prompt = translate_system("Vietnamese", "Japanese", tone="polite")
        self.assertIn("always polite", prompt)
        self.assertIn("です/ます", prompt)
        self.assertNotIn("casual speech stays casual", prompt)

    def test_casual_tone(self):
        prompt = translate_system("Vietnamese", "Japanese", tone="casual")
        self.assertIn("always casual", prompt)
        self.assertNotIn("polite speech stays polite", prompt)

    def test_unrecognized_or_empty_tone_falls_back_to_auto(self):
        auto_prompt = translate_system("Vietnamese", "Japanese", tone="auto")
        self.assertEqual(translate_system("Vietnamese", "Japanese", tone="xyz_unknown"), auto_prompt)
        self.assertEqual(translate_system("Vietnamese", "Japanese", tone=None), auto_prompt)
        self.assertEqual(translate_system("Vietnamese", "Japanese", tone=""), auto_prompt)

    def test_one_step_system_tone(self):
        polite_prompt = one_step_system("Vietnamese", "Japanese", tone="polite")
        self.assertIn("always polite", polite_prompt)

        casual_prompt = one_step_system("Vietnamese", "Japanese", tone="casual")
        self.assertIn("always casual", casual_prompt)

        auto_prompt = one_step_system("Vietnamese", "Japanese", tone="auto")
        self.assertIn("casual speech stays casual", auto_prompt)
        self.assertEqual(one_step_system("Vietnamese", "Japanese", tone="invalid"), auto_prompt)


if __name__ == "__main__":
    unittest.main()

