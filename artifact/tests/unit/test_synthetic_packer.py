import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.apkio.objects import file_sha256
from android_packer.synthetic import build_synthetic_apk
from android_packer.utils.jsonl import read_jsonl


class SyntheticPackerTests(unittest.TestCase):
    def test_xor_generation_writes_manifest_and_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload = _write_seed_apk(seed_apk)
            generated_apk = tmp / "generated.apk"
            manifest_out = tmp / "manifest.json"
            labels_out = tmp / "labels.jsonl"

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=generated_apk,
                manifest_out=manifest_out,
                labels_out=labels_out,
                transform_family="xor",
                rng_seed=7,
                xor_key=0x2A,
                enforce_payload_size_range=False,
            )

            injected = result.manifest["injected_objects"][0]
            with zipfile.ZipFile(generated_apk) as archive:
                self.assertIn("classes.dex", archive.namelist())
                transformed = archive.read(injected["object_path"])

            self.assertEqual(transformed, _xor(payload, 0x2A))
            self.assertEqual(result.manifest["generated_apk_id"], file_sha256(generated_apk))
            self.assertEqual(result.labels[0].offset_end, len(payload))
            self.assertEqual(result.labels[0].payload_sha256, file_sha256_bytes(payload))

            manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
            labels = list(read_jsonl(labels_out))
            self.assertEqual(manifest["transform_family"], "xor")
            self.assertEqual(labels[0]["object_path"], injected["object_path"])
            self.assertEqual(labels[0]["label"], "hidden_executable_payload")

    def test_split_xor_labels_cover_payload_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            payload = _write_seed_apk(seed_apk)

            result = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "generated.apk",
                transform_family="split_xor",
                rng_seed=11,
                xor_key=0x01,
                split_count=3,
                enforce_payload_size_range=False,
            )

            labels = sorted(result.labels, key=lambda label: label.part_index)
            self.assertEqual(len(labels), 3)
            self.assertEqual(labels[0].source_offset_start, 0)
            self.assertEqual(labels[-1].source_offset_end, len(payload))

            # A-v2 Fix-1 (2026-04-30): each chunk now carries an
            # independent XOR key. The first chunk still honours the
            # caller's ``xor_key=0x01`` (pins unit-test reproducibility),
            # but chunks 2..N sample fresh keys from the per-task RNG.
            # To recover the payload, we look up each segment's own key
            # in the manifest.
            injected_by_path = {
                item["object_path"]: item
                for item in result.manifest["injected_objects"]
            }
            # First segment's key must match the caller-provided value.
            first_path = labels[0].object_path
            self.assertEqual(injected_by_path[first_path]["xor_key"], 0x01)
            # Subsequent keys must be populated and not all-equal to
            # the caller-provided key (the whole point of the fix).
            keys = [injected_by_path[lb.object_path]["xor_key"] for lb in labels]
            self.assertTrue(all(k is not None for k in keys))
            self.assertGreater(
                len({k for k in keys}), 1,
                "A-v2 Fix-1: split_xor must use independent per-segment keys",
            )

            recovered = []
            with zipfile.ZipFile(result.generated_apk_path) as archive:
                for label in labels:
                    key = injected_by_path[label.object_path]["xor_key"]
                    recovered.append(_xor(archive.read(label.object_path), key))
            self.assertEqual(b"".join(recovered), payload)

    def test_generation_is_deterministic_for_same_seed_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed_apk = tmp / "seed.apk"
            _write_seed_apk(seed_apk)

            first = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "first.apk",
                transform_family="xor",
                rng_seed=19,
                xor_key=0x05,
                enforce_payload_size_range=False,
            )
            second = build_synthetic_apk(
                seed_apk=seed_apk,
                generated_apk_out=tmp / "second.apk",
                transform_family="xor",
                rng_seed=19,
                xor_key=0x05,
                enforce_payload_size_range=False,
            )

            self.assertEqual(
                file_sha256(first.generated_apk_path),
                file_sha256(second.generated_apk_path),
            )
            self.assertEqual(first.labels[0].object_path, second.labels[0].object_path)


def _write_seed_apk(path: Path) -> bytes:
    payload = b"dex\n035\x00synthetic-payload-bytes"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.dex", payload)
        archive.writestr("assets/readme.txt", b"benign")
    return payload


def _xor(data: bytes, key: int) -> bytes:
    return bytes(value ^ key for value in data)


def file_sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    unittest.main()
