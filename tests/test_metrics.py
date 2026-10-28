import unittest

from app.metrics import cer, metric_for, wer


class WerTests(unittest.TestCase):
    def test_identical_ignoring_case_and_punctuation(self):
        self.assertEqual(wer("Xin chào, bạn!", "xin chào bạn"), 0.0)

    def test_one_substitution_in_four_words(self):
        self.assertAlmostEqual(wer("tôi là sinh viên", "tôi là giáo viên"), 0.25)

    def test_deletion_and_insertion(self):
        self.assertAlmostEqual(wer("a b c d", "a c d"), 0.25)
        self.assertAlmostEqual(wer("a b", "a b c d"), 1.0)

    def test_empty_reference(self):
        self.assertEqual(wer("", ""), 0.0)
        self.assertEqual(wer("", "abc"), 1.0)


class CerTests(unittest.TestCase):
    def test_japanese_one_char_error_cer_vs_wer(self):
        # 10 ký tự tiếng Nhật: câu đúng vs câu sai 1 ký tự
        ref = "あいうえおかきくけこ"  # 10 ký tự
        hyp = "あいうえおかきくけさ"  # sai 1 ký tự cuối

        # CER phản ánh đúng tỉ lệ lỗi: 1 / 10 = 0.1
        self.assertAlmostEqual(cer(ref, hyp), 0.1)

        # WER coi cả câu liền mạch là 1 từ duy nhất, nên sai 1 ký tự trả về 1.0 (100% lỗi)
        self.assertEqual(wer(ref, hyp), 1.0)

    def test_cer_halfwidth_fullwidth_punctuation(self):
        # NFKC chuẩn hóa toàn/bán chiều và bỏ dấu câu
        ref = "今日は、いい天気ですね！"
        hyp = "今日はいい天気ですね"
        self.assertEqual(cer(ref, hyp), 0.0)

    def test_empty_reference(self):
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "あ"), 1.0)


class MetricForTests(unittest.TestCase):
    def test_metric_selection(self):
        name_ja, fn_ja = metric_for("Japanese")
        self.assertEqual(name_ja, "CER")
        self.assertIs(fn_ja, cer)

        name_vi, fn_vi = metric_for("Vietnamese")
        self.assertEqual(name_vi, "WER")
        self.assertIs(fn_vi, wer)


if __name__ == "__main__":
    unittest.main()
