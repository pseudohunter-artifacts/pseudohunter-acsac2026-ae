import tempfile
import unittest
import zipfile
from pathlib import Path

from android_packer.apkio.objects import classify_object, iter_apk_objects


class ApkIoTests(unittest.TestCase):
    def test_classify_common_apk_objects(self):
        self.assertEqual(classify_object("classes.dex", b"dex\n035\x00"), "dex")
        self.assertEqual(classify_object("assets/payload.bin", b"\x01\x02"), "asset_blob")
        self.assertEqual(classify_object("lib/arm64-v8a/libx.so", b"\x7fELF"), "native_lib")
        self.assertEqual(classify_object("res/raw/logo.png", b"\x89PNG"), "resource")
        self.assertEqual(classify_object("META-INF/CERT.RSA", b""), "signature")

    def test_iter_apk_objects_reads_zip_members(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apk_path = Path(tmpdir) / "sample.apk"
            with zipfile.ZipFile(apk_path, "w") as archive:
                archive.writestr("classes.dex", b"dex\n035\x00")
                archive.writestr("assets/payload.bin", b"payload")

            objects = list(iter_apk_objects(apk_path))
            paths = [metadata.object_path for metadata, _ in objects]
            types = [metadata.object_type for metadata, _ in objects]

            self.assertEqual(paths, ["classes.dex", "assets/payload.bin"])
            self.assertEqual(types, ["dex", "asset_blob"])


if __name__ == "__main__":
    unittest.main()
