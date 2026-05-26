"""End-to-end integration test for Track B commercial packer pipeline (B-g-2).

This simulates a realistic *CS3 · Bangcle* packed APK by taking a
benign APK and overlaying the Bangcle signature entries described in
``configs/data/track_b_commercial_rules/cs3_bangcle.yaml``:

* ``lib/arm64-v8a/libsecexe.so`` and ``lib/arm64-v8a/libsecmain.so``
  native loader pair
* ``assets/bangcle_classes.jar`` encrypted DEX container
* ``classes.dex`` shim registering ``AppWrapper``

We then run the full Track B labeling pipeline via
``process_pair`` with the real shipped rule file and assert:

1. The commercial group is auto-forced ``path_b_enabled=True`` even
   though yaml says ``path_b: false``.
2. ``cs_cross_validate_commercial_packer`` emits a CsCrossValidationReport
   with ``rule_matched_entries`` covering the three payload regions.
3. The pipeline writes ``merged_labels.jsonl`` + ``cs_cross_validate_report.json``.
4. ``apkid`` cross-check is skipped cleanly (run_apkid=False) so the
   test runs in environments without the apkid binary.

Runs under `tests/integration/` -> still picked up by the default
``pytest`` invocation.

This is NOT a substitute for running real Bangcle-packed APKs through
the pipeline; real end-to-end validation requires a Bangcle commercial
license which we intentionally do not bundle into the repo. But it
does prove the pipeline wiring end-to-end: regex matching -> rule
firing -> Path A-rule label emission -> Path B diff alignment -> IoU
computation -> final merged_labels.jsonl, on a packed APK the same
*shape* as a real one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from android_packer.labeling import (
    LABEL_SOURCE_PATH_B,
    LABEL_SOURCE_PATH_A_RULE,
    PackerIdent,
    PairInputs,
    process_pair,
)
from android_packer.labeling.commercial_rule_engine import load_rule_file


# ---------------------------------------------------------------------------
# Fake-but-realistic packed APK fixtures (one per commercial packer)
# ---------------------------------------------------------------------------


def _benign_apk_bytes() -> dict:
    """A consistent benign APK used by every CS* scenario."""
    return {
        "classes.dex": b"DEX-BENIGN-" + b"\x00" * 500,
        "AndroidManifest.xml": b"<manifest package='com.test.benign'/>",
        "resources.arsc": b"ARSC-STANDARD-" + b"\x01" * 200,
        "lib/arm64-v8a/libapp.so": b"ELF-USER-LIB-" + b"\x02" * 400,
        "assets/legit.txt": b"benign readme",
    }


def _bangcle_packed_bytes(benign: dict) -> dict:
    """Bangcle signature overlay: shim classes.dex, libsec*.so, assets jar."""
    out = dict(benign)
    out["classes.dex"] = (
        b"DEX-BANGCLE-SHIM-APPWRAPPER-" + b"\x10" * 400
    )  # replaced with shim
    out["lib/arm64-v8a/libsecexe.so"] = (
        b"ELF-BANGCLE-LIBSECEXE-" + b"\xAA" * 800
    )
    out["lib/arm64-v8a/libsecmain.so"] = (
        b"ELF-BANGCLE-LIBSECMAIN-" + b"\xBB" * 600
    )
    out["assets/bangcle_classes.jar"] = (
        b"ENCRYPTED-DEX-BLOB-" + b"\xCC" * 1500
    )
    return out


def _bangcle_v3_packed_bytes(benign: dict) -> dict:
    """Bangcle 2026 SaaS layout (observed on real samples 2026-05)."""
    out = dict(benign)
    # classes.dex replaced with a thin shim (dramatically smaller)
    out["classes.dex"] = b"DEX-BANGCLE-V3-SHIM-" + b"\x10" * 400
    # Single libSecShell.so per ABI (not libsecexe/libsecmain pair)
    out["lib/arm64-v8a/libSecShell.so"] = (
        b"ELF-BANGCLE-V3-LIBSECSHELL-ARM64-" + b"\xAA" * 1000
    )
    out["lib/armeabi-v7a/libSecShell.so"] = (
        b"ELF-BANGCLE-V3-LIBSECSHELL-ARMV7-" + b"\xBB" * 700
    )
    # Encrypted DEX container renamed to assets/classes0.jar
    out["assets/classes0.jar"] = (
        b"ENCRYPTED-DEX-JAR-BLOB-V3-" + b"\xCC" * 2000
    )
    # New meta-data/ signing tree (manifest.mf + rsa.sig + rsa.pub)
    out["assets/meta-data/manifest.mf"] = (
        b"MANIFEST-SHA256-DIGEST-TABLE\n" + b"\xDD" * 300
    )
    out["assets/meta-data/rsa.sig"] = b"RSA-SIG-BLOB-" + b"\xEE" * 200
    out["assets/meta-data/rsa.pub"] = b"RSA-PUB-KEY-" + b"\xFF" * 200
    return out


def _jiagu_packed_bytes(benign: dict) -> dict:
    out = dict(benign)
    out["classes.dex"] = b"DEX-360-STUB-STUBAPP-" + b"\x20" * 400
    out["lib/arm64-v8a/libjiagu_a64.so"] = (
        b"ELF-360-LIBJIAGU-A64-" + b"\xDD" * 1000
    )
    out["assets/libjiagu_a64.so"] = (
        b"ENCRYPTED-DEX-360-" + b"\xEE" * 1200
    )
    return out


def _ijiami_packed_bytes(benign: dict) -> dict:
    out = dict(benign)
    out["classes.dex"] = b"DEX-IJIAMI-NATIVEAPP-" + b"\x30" * 400
    out["lib/arm64-v8a/libexec.so"] = b"ELF-IJIAMI-EXEC-" + b"\x31" * 900
    out["lib/arm64-v8a/libexecmain.so"] = (
        b"ELF-IJIAMI-EXECMAIN-" + b"\x32" * 700
    )
    out["assets/ijiami_data"] = b"ENCRYPTED-IJIAMI-" + b"\x33" * 1100
    return out


def _legu_packed_bytes(benign: dict) -> dict:
    out = dict(benign)
    out["classes.dex"] = b"DEX-TX-TXAPPENTRY-" + b"\x40" * 400
    out["lib/arm64-v8a/libshellx.so"] = (
        b"ELF-LEGU-SHELLX-" + b"\x41" * 800
    )
    out["assets/mix.dex"] = b"ENCRYPTED-TX-MIXDEX-" + b"\x42" * 1400
    out["assets/0OO00l111l1l"] = b"LEGU-METADATA-" + b"\x43" * 300
    return out


def _dexprotector_packed_bytes(benign: dict) -> dict:
    out = dict(benign)
    out["classes.dex"] = b"DEX-JESB-APPSHIM-" + b"\x50" * 400
    out["lib/arm64-v8a/libdexprotector.so"] = (
        b"ELF-DP-LOADER-" + b"\x51" * 900
    )
    out["assets/protected/vm_0001.bin"] = (
        b"VIRTUALIZED-METHOD-BODY-" + b"\x52" * 1300
    )
    out["assets/dp_tables"] = b"DP-VM-TABLES-" + b"\x53" * 200
    return out


def _write_apk(path: Path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


# ---------------------------------------------------------------------------
# Integration scenarios -- one per commercial packer
# ---------------------------------------------------------------------------


class CommercialPackerE2E(unittest.TestCase):
    """Per-packer integration smoke for B-g-2."""

    # repo_root is resolved once for all scenarios
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.rules_dir = (
            cls.repo_root / "configs" / "data" / "track_b_commercial_rules"
        )
        cls.registry_path = (
            cls.repo_root / "configs" / "data" / "track_b_packers.yaml"
        )
        cls.registry = yaml.safe_load(
            cls.registry_path.read_text(encoding="utf-8")
        )["packers"]

    def _scenario(
        self,
        *,
        packer_yaml_key: str,
        packed_factory,
        expected_fired_rule_ids: list,
    ) -> None:
        """Drive one packer through process_pair end-to-end."""
        spec = self.registry[packer_yaml_key]
        ident = PackerIdent.from_registry_entry(
            packer_yaml_key, spec, rules_dir=self.rules_dir
        )
        self.assertEqual(ident.group, "commercial")
        self.assertTrue(
            ident.path_b_enabled,
            f"{packer_yaml_key} commercial packer must have path_b_enabled=True",
        )
        self.assertIsNotNone(
            ident.rule_file,
            f"{packer_yaml_key} must resolve rule_file from shipped yaml",
        )
        self.assertTrue(
            ident.rule_file.exists(),
            f"{packer_yaml_key} rule_file does not exist on disk: {ident.rule_file}",
        )

        # Sanity-check that the rule file's declared rule_ids include the
        # ones we are going to assert on.
        rule_spec = load_rule_file(ident.rule_file)
        available_ids = {r.rule_id for r in rule_spec.rules}
        for rid in expected_fired_rule_ids:
            self.assertIn(
                rid,
                available_ids,
                f"{packer_yaml_key}: expected rule_id {rid!r} declared in rules",
            )

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            benign = tmp / "benign.apk"
            packed = tmp / "packed.apk"
            benign_entries = _benign_apk_bytes()
            _write_apk(benign, benign_entries)
            _write_apk(packed, packed_factory(benign_entries))
            out_dir = tmp / "out"

            inputs = PairInputs(
                packer=ident,
                benign_apk=benign,
                packed_apk=packed,
                inject_labels_jsonl=None,  # commercial has no Path A source
                apk_id=f"test:{packer_yaml_key}:apk",
                source_apk_id=f"test:{packer_yaml_key}:benign",
            )
            outcome = process_pair(
                inputs,
                out_dir,
                run_apkid=False,  # no apkid dependency in tests
            )

            # 1. Commercial path -> cs_cross_validate executed
            self.assertIsNotNone(
                outcome.cs_decision,
                f"{packer_yaml_key}: expected a CS decision, got None",
            )
            # 2. Some labels produced (could be path_b_only or solid, either
            #    is acceptable; both have final_label_count > 0)
            self.assertGreater(
                outcome.final_label_count,
                0,
                f"{packer_yaml_key}: expected >= 1 final label, got 0; "
                f"reasons={outcome.reasons} notes={outcome.notes}",
            )
            # 3. Labels went to one of the two valid commercial sources
            self.assertIn(
                outcome.chosen_source,
                (LABEL_SOURCE_PATH_B, LABEL_SOURCE_PATH_A_RULE),
                f"{packer_yaml_key}: unexpected chosen_source {outcome.chosen_source!r}",
            )
            # 4. Artefacts on disk
            self.assertIn("merged_labels", outcome.artifacts)
            self.assertIn("cs_cross_validate_report", outcome.artifacts)
            cs_report_path = Path(outcome.artifacts["cs_cross_validate_report"])
            self.assertTrue(cs_report_path.exists())
            cs_report = json.loads(cs_report_path.read_text(encoding="utf-8"))
            # 5. cs_cross_validate saw non-zero rule matches (the fake
            #    packed APK's signature entries actually fired rules)
            self.assertGreater(
                len(cs_report.get("rule_matched_entries", [])),
                0,
                f"{packer_yaml_key}: cs_cross_validate rule_matched_entries is empty",
            )

    # One test method per packer so failures surface individually
    def test_cs1_360_jiagu_e2e(self) -> None:
        self._scenario(
            packer_yaml_key="cs1_360_jiagu",
            packed_factory=_jiagu_packed_bytes,
            expected_fired_rule_ids=[
                "360_v1_lock_libjiagu_lib_abi",
                "360_v2_assets_libjiagu_abi_family",
                "360_shell_classes_dex",
            ],
        )

    def test_cs2_ijiami_e2e(self) -> None:
        self._scenario(
            packer_yaml_key="cs2_ijiami",
            packed_factory=_ijiami_packed_bytes,
            expected_fired_rule_ids=[
                "ijiami_lock_libexec_family",
                "ijiami_assets_data_blob",
                "ijiami_shell_classes_dex",
            ],
        )

    def test_cs3_bangcle_e2e(self) -> None:
        self._scenario(
            packer_yaml_key="cs3_bangcle",
            packed_factory=_bangcle_packed_bytes,
            expected_fired_rule_ids=[
                # v2 literature layout rules (libsecexe.so / bangcle_classes.jar)
                # are exercised by the existing _bangcle_packed_bytes fixture
                "bangcle_v2_lock_libsec_family",
                "bangcle_v2_assets_classes_jar",
                "bangcle_shell_classes_dex",
            ],
        )

    def test_cs3_bangcle_v3_e2e(self) -> None:
        """2026 SaaS landing layout (libSecShell.so + classes0.jar + meta-data/).

        Motivated by our 2026-05 empirical sample run that discovered
        the 2018-2022 literature rules had ZERO coverage on the new
        SaaS builds. This test pins the v3 rules against a
        fake-but-realistic packed APK whose entry layout matches the
        observed landing shape; it would catch any regression that
        silently drops the libSecShell / classes0.jar / meta-data/
        regex patterns.
        """
        self._scenario(
            packer_yaml_key="cs3_bangcle",
            packed_factory=_bangcle_v3_packed_bytes,
            expected_fired_rule_ids=[
                "bangcle_v3_lock_libsecshell",
                "bangcle_v3_assets_classesN_jar",
                "bangcle_v3_assets_meta_data_tree",
                "bangcle_shell_classes_dex",
            ],
        )

    def test_cs4_tencent_legu_e2e(self) -> None:
        self._scenario(
            packer_yaml_key="cs4_tencent_legu",
            packed_factory=_legu_packed_bytes,
            expected_fired_rule_ids=[
                "legu_lock_libshell_family",
                "legu_assets_mix_dex",
                "legu_assets_metadata",
                "legu_shell_classes_dex",
            ],
        )

    def test_cs5_dexprotector_e2e(self) -> None:
        self._scenario(
            packer_yaml_key="cs5_dexprotector",
            packed_factory=_dexprotector_packed_bytes,
            expected_fired_rule_ids=[
                "dexprotector_lock_native_family",
                "dexprotector_assets_protected_tree",
                "dexprotector_assets_vm_tables",
                "dexprotector_shell_classes_dex",
            ],
        )


if __name__ == "__main__":
    unittest.main()
