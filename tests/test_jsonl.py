import json
import tempfile
import unittest
from pathlib import Path
from src.common import append_jsonl, read_jsonl

class JsonlTests(unittest.TestCase):
    def test_unicode_separators_are_not_record_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [{"completion": "before" + chr(code) + "after"}
                    for code in (0x85, 0x2028, 0x2029)]
            for row in rows:
                append_jsonl(path, row)
            self.assertEqual(read_jsonl(path), rows)

    def test_corrupt_record_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(chr(123) + chr(34) + "unfinished", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_jsonl(path)
