import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench import (
    build_output_filename,
    calculate_percentile,
    compute_bench_summary,
    lang_code,
    main,
)


class LangCodeTests(unittest.TestCase):
    def test_known_languages(self):
        self.assertEqual(lang_code("Vietnamese"), "vi")
        self.assertEqual(lang_code("vietnamese"), "vi")
        self.assertEqual(lang_code("vi"), "vi")
        self.assertEqual(lang_code("VI"), "vi")
        self.assertEqual(lang_code("Japanese"), "ja")
        self.assertEqual(lang_code("japanese"), "ja")
        self.assertEqual(lang_code("ja"), "ja")
        self.assertEqual(lang_code("English"), "en")

    def test_fallback_code(self):
        self.assertEqual(lang_code("korean"), "ko")
        self.assertEqual(lang_code(""), "")


class OutputFilenameTests(unittest.TestCase):
    def test_default_mode_and_tag(self):
        # Default mode=None -> two_step; tag=groq
        fn = build_output_filename(None, "Vietnamese", "Japanese", "groq")
        self.assertEqual(fn, "bench_two_step_vi2ja_groq.csv")

    def test_custom_mode_and_reverse_direction(self):
        fn = build_output_filename("one_step", "Japanese", "Vietnamese", "gemini")
        self.assertEqual(fn, "bench_one_step_ja2vi_gemini.csv")

    def test_without_tag(self):
        fn = build_output_filename("two_step", "vi", "ja", "")
        self.assertEqual(fn, "bench_two_step_vi2ja.csv")

    def test_tag_with_leading_underscore_no_double_underscore(self):
        fn = build_output_filename(None, "Vietnamese", "Japanese", "_groq")
        self.assertEqual(fn, "bench_two_step_vi2ja_groq.csv")


class PercentileTests(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(calculate_percentile([], 90), 0.0)

    def test_single_element(self):
        self.assertEqual(calculate_percentile([42.0], 50), 42.0)
        self.assertEqual(calculate_percentile([42.0], 90), 42.0)

    def test_multi_elements_p90_and_median(self):
        # Sequence 1 to 10
        vals = [float(i) for i in range(1, 11)]
        p50 = calculate_percentile(vals, 50)
        p90 = calculate_percentile(vals, 90)
        self.assertAlmostEqual(p50, 5.5)
        self.assertAlmostEqual(p90, 9.1)


class BenchSummaryTests(unittest.TestCase):
    def test_empty_ref_skips_metric_but_keeps_timings_and_slowest(self):
        rows = [
            # Clip with reference
            {
                "file": "vi_16.m4a", "size_kb": 92.2, "metric": "WER", "score": 0.10, "loi": "",
                "giay": 1.2, "asr_ms": 600, "mt_ms": 500, "nguyen_van": "Xin chào",
            },
            {
                "file": "vi_17.m4a", "size_kb": 168.1, "metric": "WER", "score": 0.20, "loi": "",
                "giay": 1.8, "asr_ms": 800, "mt_ms": 700, "nguyen_van": "Tạm biệt",
            },
            # Clip WITHOUT reference (nguyen_van is empty)
            {
                "file": "vi_01.m4a", "size_kb": 45.0, "metric": "WER", "score": "", "loi": "",
                "giay": 1.5, "asr_ms": 700, "mt_ms": 600, "nguyen_van": "",
            },
            {
                "file": "vi_04.m4a", "size_kb": 96.0, "metric": "WER", "score": "", "loi": "",
                "giay": 2.2, "asr_ms": 1100, "mt_ms": 650, "nguyen_van": "",
            },
            # Failed clip
            {
                "file": "vi_02.m4a", "size_kb": 64.0, "metric": "WER", "score": "", "loi": "API timeout",
                "giay": 3.0, "asr_ms": 0, "mt_ms": 0, "nguyen_van": "",
            },
        ]

        summary = compute_bench_summary(rows, "WER")

        self.assertEqual(summary["total_files"], 5)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["success_count"], 4)

        # Accuracy metric only computed on 2 clips with references
        self.assertEqual(summary["n_scored"], 2)
        self.assertAlmostEqual(summary["avg_score"], 0.15)
        self.assertTrue(summary["warning_low_n"])  # n=2 < 20

        # Latency computed across all 4 successful clips
        # asr_ms: [600, 700, 800, 1100] -> median (700+800)/2 = 750
        self.assertEqual(summary["asr_median_ms"], 750)
        # mt_ms: [500, 600, 650, 700] -> median (600+650)/2 = 625
        self.assertEqual(summary["mt_median_ms"], 625)

        # Top 3 slowest files sorted by ASR: 1100 (vi_04), 800 (vi_17), 700 (vi_01)
        slowest = summary["slowest_files"]
        self.assertEqual(len(slowest), 3)
        self.assertEqual(slowest[0]["file"], "vi_04.m4a")
        self.assertEqual(slowest[0]["asr_ms"], 1100)
        self.assertAlmostEqual(slowest[0]["size_kb"], 96.0)

        self.assertEqual(slowest[1]["file"], "vi_17.m4a")
        self.assertEqual(slowest[1]["asr_ms"], 800)
        self.assertAlmostEqual(slowest[1]["size_kb"], 168.1)

        self.assertEqual(slowest[2]["file"], "vi_01.m4a")
        self.assertEqual(slowest[2]["asr_ms"], 700)
        self.assertAlmostEqual(slowest[2]["size_kb"], 45.0)

    def test_all_refs_empty_still_computes_latency_and_slowest(self):
        rows = [
            {"file": "ja_01.m4a", "size_kb": 24.5, "score": "", "loi": "", "giay": 1.0, "asr_ms": 400, "mt_ms": 400, "nguyen_van": ""},
            {"file": "ja_02.m4a", "size_kb": 23.5, "score": "", "loi": "", "giay": 1.4, "asr_ms": 500, "mt_ms": 600, "nguyen_van": ""},
        ]
        summary = compute_bench_summary(rows, "CER")
        self.assertEqual(summary["n_scored"], 0)
        self.assertIsNone(summary["avg_score"])
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        # Latency (b) still computed completely
        self.assertEqual(summary["asr_median_ms"], 450)
        self.assertEqual(summary["mt_median_ms"], 500)
        self.assertEqual(len(summary["slowest_files"]), 2)
        self.assertEqual(summary["slowest_files"][0]["file"], "ja_02.m4a")
        self.assertEqual(summary["slowest_files"][0]["asr_ms"], 500)
        self.assertAlmostEqual(summary["slowest_files"][0]["size_kb"], 23.5)


class MainExecutionTests(unittest.TestCase):
    def test_main_execution_generates_correct_csv_and_prints_section_b(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_dir = tmp_path / "audio"
            audio_dir.mkdir()

            f1 = audio_dir / "vi_01.m4a"
            f1.write_bytes(b"\x00" * 2048)  # 2.0 KB
            f2 = audio_dir / "vi_16.m4a"
            f2.write_bytes(b"\x00" * 4096)  # 4.0 KB

            refs_data = {
                "vi_01.m4a": "",  # không có bản gốc
                "vi_16.m4a": "Xin chào",  # có bản gốc
            }
            refs_file = tmp_path / "refs.json"
            refs_file.write_text(json.dumps(refs_data, ensure_ascii=False), encoding="utf-8")

            mock_interp = MagicMock()
            mock_res1 = MagicMock(original="Chào bạn", translated="こんにちは", timings_ms={"asr": 320, "mt": 210})
            mock_res2 = MagicMock(original="Xin chào", translated="こんにちは", timings_ms={"asr": 540, "mt": 280})
            mock_interp.interpret.side_effect = [mock_res1, mock_res2]

            csv_out = tmp_path / "bench_two_step_vi2ja_groq.csv"

            test_argv = [
                "bench.py",
                str(refs_file),
                str(audio_dir),
                "--src", "Vietnamese",
                "--dst", "Japanese",
                "--tag", "groq",
                "--pace", "1.5",
            ]

            out_buf = io.StringIO()
            with patch("sys.argv", test_argv), \
                 patch("bench.build_interpreter", return_value=mock_interp), \
                 patch("bench.build_output_filename", return_value=str(csv_out)), \
                 patch("time.sleep") as mock_sleep, \
                 patch("sys.stdout", out_buf):
                main()

            output = out_buf.getvalue()

            # Verify pace sleep was called once between 2 files (file 1 -> sleep -> file 2, no sleep after file 2)
            self.assertEqual(mock_sleep.call_count, 1)
            mock_sleep.assert_called_with(1.5)

            # Verify section (b) and summary are printed completely
            self.assertIn("BENCHMARK RESULTS (Vietnamese -> Japanese)", output)
            self.assertIn("(a) RECOGNITION ACCURACY", output)
            self.assertIn("(b) PROCESSING LATENCY (ALL SUCCESSFUL FILES):", output)
            self.assertIn("Speech Recognition (ASR) : median 430 ms", output)
            self.assertIn("Translation (MT)         : median 245 ms", output)
            self.assertIn("Top 3 slowest files:", output)
            self.assertIn("vi_16.m4a: asr_ms=540 ms, size_kb=4.0 KB", output)
            self.assertIn("vi_01.m4a: asr_ms=320 ms, size_kb=2.0 KB", output)

            # Kiểm tra nội dung CSV
            self.assertTrue(csv_out.exists())
            with open(csv_out, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertIn("size_kb", reader.fieldnames)
                self.assertEqual(rows[0]["file"], "vi_01.m4a")
                self.assertEqual(rows[0]["size_kb"], "2.0")
                self.assertEqual(rows[0]["score"], "")  # ref rỗng
                self.assertEqual(rows[1]["file"], "vi_16.m4a")
                self.assertEqual(rows[1]["size_kb"], "4.0")
                self.assertNotEqual(rows[1]["score"], "")


if __name__ == "__main__":
    unittest.main()

