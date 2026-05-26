"""Contract tests: runtime artefacts must validate against their schemas."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.apkio import iter_apk_objects
from android_packer.labeling import build_training_labels
from android_packer.regioning import iter_regions
from android_packer.synthetic import build_synthetic_apk
from android_packer.utils.schema import validate_record, validate_records


def _write_seed_apk(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.dex", b"dex\n035\x00" + (b"payload bytes " * 32))
        archive.writestr("assets/readme.txt", b"benign text for the window")


class PipelineArtifactContractTests(unittest.TestCase):
    def test_region_metadata_matches_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            apk = tmp_path / "seed.apk"
            _write_seed_apk(apk)

            regions = []
            for metadata, data in iter_apk_objects(apk):
                regions.extend(
                    region.to_dict()
                    for region in iter_regions(metadata, data, window_size=16, stride=16)
                )
            self.assertGreater(len(regions), 0)
            validate_records(regions, "region_metadata")

    def test_synthetic_manifest_and_labels_match_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            apk = tmp_path / "seed.apk"
            _write_seed_apk(apk)

            result = build_synthetic_apk(
                seed_apk=apk,
                generated_apk_out=tmp_path / "generated.apk",
                transform_family="xor",
                rng_seed=42,
                enforce_payload_size_range=False,
            )

            validate_record(result.manifest, "synthetic_manifest")
            validate_records(
                (label.to_dict() for label in result.labels),
                "synthetic_label",
            )

    def test_training_labels_match_region_training_label_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            apk = tmp_path / "seed.apk"
            _write_seed_apk(apk)
            synthetic = build_synthetic_apk(
                seed_apk=apk,
                generated_apk_out=tmp_path / "generated.apk",
                transform_family="xor",
                rng_seed=42,
                enforce_payload_size_range=False,
            )
            # Re-read regions from the generated APK so labels align with
            # real object paths produced by the packer.
            regions = []
            for metadata, data in iter_apk_objects(synthetic.generated_apk_path):
                regions.extend(
                    region.to_dict()
                    for region in iter_regions(metadata, data, window_size=16, stride=16)
                )
            labels = build_training_labels(
                regions=regions,
                synthetic_labels=[label.to_dict() for label in synthetic.labels],
            )
            validate_records(
                (row.to_dict() for row in labels.region_labels),
                "region_training_label",
            )


if __name__ == "__main__":
    unittest.main()
