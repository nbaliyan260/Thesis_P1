"""Offline standard-library regression tests; all mutations use temporary fixtures."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


SPEC = importlib.util.spec_from_file_location("verify_phases", Path(__file__).with_name("verify_phases.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def record(name, data):
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recut-phase-verifier-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_json(self, name, value):
        return self.write(name, json.dumps(value, indent=2).encode())

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments], stderr=subprocess.PIPE)

    def commit(self):
        self.git("add", "--all")
        self.git("-c", "user.name=Offline Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD").decode().strip()

    def history(self, records=None, symlink=False):
        self.git("init", "-q")
        if symlink:
            self.write("target", b"original")
            (self.root / "payload").symlink_to("target")
        else:
            self.write("payload", b"original")
        self.write_json(verifier.PHASE1_MANIFEST, {"records": records or [record("payload", b"original")]})
        return self.commit()

    def phase2(self):
        self.payloads = {"source.py": b"print('example')\n", "snapshots/state.pt": b"tensor bytes fixture"}
        self.delivery = {"records": [record(name, data) for name, data in self.payloads.items()]}
        manifest = self.write_json(verifier.DELIVERY_MANIFEST, self.delivery)
        self.write("source.py", self.payloads["source.py"])
        self.repository = {"phase": "phase-2", "external_artifact": dict(verifier.EXTERNAL_ARTIFACT),
                           "records": [record("source.py", self.payloads["source.py"]),
                                       record(verifier.DELIVERY_MANIFEST, manifest.read_bytes())]}
        self.write_json(verifier.PHASE2_MANIFEST, self.repository)
        return verifier.verify_phase2(self.root)[0]

    def zip_fixture(self, members=None):
        if members is None:
            members = list(self.payloads.items()) + [(verifier.DELIVERY_MANIFEST,
                                                    (self.root / verifier.DELIVERY_MANIFEST).read_bytes())]
        path = self.root / "artifact.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as stream:
                for name, data in members:
                    stream.writestr(name, data)
        return path, record(path.name, path.read_bytes())


class PathAndJsonTests(Fixtures):
    def test_reject_unsafe_paths(self):
        for name in ("", "/tmp/file", "../file", "a/../b", "a/./b", "./file", "a//b", "a/",
                     "a\\b", "C:/file", "a\x00b", "a\nb", "a\x7fb", None, 7):
            with self.subTest(name=repr(name)), self.assertRaises(verifier.VerificationError):
                verifier.safe_name(name)

    def test_spaces_and_unicode_are_normal_paths(self):
        self.assertEqual(verifier.safe_name("Research Papers PDFs/résumé.pdf"), "Research Papers PDFs/résumé.pdf")

    def test_reject_duplicate_json_keys_and_nonfinite(self):
        for data in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{'):
            with self.subTest(data=data), self.assertRaises(verifier.VerificationError):
                verifier.parse_json(data)

    def test_reject_invalid_record_schema(self):
        good = record("a", b"a")
        invalid = ([], [None], [good, good], [{**good, "bytes": True}], [{**good, "bytes": -1}],
                   [{**good, "sha256": "z" * 64}], [{**good, "sha256": "A" * 64}],
                   [{**good, "extra": 1}], [{"path": "a"}])
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(verifier.VerificationError):
                verifier.record_map(records, "fixture")

    def test_reject_symlink_file(self):
        self.write("target", b"value")
        (self.root / "link").symlink_to("target")
        with self.assertRaisesRegex(verifier.VerificationError, "Symlink"):
            verifier.file_path(self.root, "link")

    def test_reject_symlink_parent_even_inside_repository(self):
        self.write("actual/file", b"value")
        (self.root / "alias").symlink_to("actual", target_is_directory=True)
        with self.assertRaisesRegex(verifier.VerificationError, "Symlink"):
            verifier.file_path(self.root, "alias/file")

    def test_reject_symlink_parent_outside_repository(self):
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "file").write_bytes(b"value")
            (self.root / "alias").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(verifier.VerificationError, "Symlink"):
                verifier.file_path(self.root, "alias/file")

    def test_reject_directory_as_file(self):
        (self.root / "directory").mkdir()
        with self.assertRaisesRegex(verifier.VerificationError, "Missing regular file"):
            verifier.file_path(self.root, "directory")


class HistoryTests(Fixtures):
    def test_historical_blobs_ignore_current_changes_and_do_not_mutate_checkout(self):
        commit = self.history()
        self.write("payload", b"changed current phase")
        self.write_json(verifier.PHASE1_MANIFEST, {"records": []})
        before = self.git("status", "--porcelain=v1", "-z")
        self.assertEqual(verifier.verify_phase1(self.root, commit), 1)
        self.assertEqual(before, self.git("status", "--porcelain=v1", "-z"))
        self.assertEqual((self.root / "payload").read_bytes(), b"changed current phase")

    def test_historical_manifest_tamper(self):
        commit = self.history([record("payload", b"not what was committed")])
        with self.assertRaisesRegex(verifier.VerificationError, "mismatch"):
            verifier.verify_phase1(self.root, commit)

    def test_missing_historical_file(self):
        commit = self.history([record("missing", b"not present")])
        with self.assertRaisesRegex(verifier.VerificationError, "Missing historical"):
            verifier.verify_phase1(self.root, commit)

    def test_historical_symlink_rejected(self):
        commit = self.history(symlink=True)
        with self.assertRaisesRegex(verifier.VerificationError, "Non-regular historical"):
            verifier.verify_phase1(self.root, commit)

    def test_missing_commit_diagnostic(self):
        self.history()
        with self.assertRaisesRegex(verifier.VerificationError, "shallow clone"):
            verifier.verify_phase1(self.root, "0" * 40)


class RepositoryTests(Fixtures):
    def test_current_manifest_success_without_snapshot_files(self):
        records = self.phase2()
        self.assertEqual(len(records), 2)
        self.assertFalse((self.root / "snapshots/state.pt").exists())

    def test_current_same_length_tamper(self):
        self.phase2()
        self.write("source.py", b"X" * len(self.payloads["source.py"]))
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            verifier.verify_phase2(self.root)

    def test_current_size_tamper(self):
        self.phase2()
        self.write("source.py", b"longer source file with changed contents")
        with self.assertRaisesRegex(verifier.VerificationError, "Size mismatch"):
            verifier.verify_phase2(self.root)

    def test_missing_current_file(self):
        self.phase2()
        (self.root / "source.py").unlink()
        with self.assertRaisesRegex(verifier.VerificationError, "Missing regular file"):
            verifier.verify_phase2(self.root)

    def test_current_delivery_manifest_must_be_bound(self):
        self.phase2()
        self.repository["records"] = self.repository["records"][:1]
        self.write_json(verifier.PHASE2_MANIFEST, self.repository)
        with self.assertRaisesRegex(verifier.VerificationError, "must bind"):
            verifier.verify_phase2(self.root)

    def test_external_artifact_binding_cannot_change(self):
        self.phase2()
        self.repository["external_artifact"]["url"] = "https://example.invalid/not-the-release.zip"
        self.write_json(verifier.PHASE2_MANIFEST, self.repository)
        with self.assertRaisesRegex(verifier.VerificationError, "external artifact"):
            verifier.verify_phase2(self.root)

    def test_current_duplicate_record(self):
        self.phase2()
        self.repository["records"].append(self.repository["records"][0])
        self.write_json(verifier.PHASE2_MANIFEST, self.repository)
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate path"):
            verifier.verify_phase2(self.root)


class ArchiveTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.records = self.phase2()

    def verify(self, members=None, **kwargs):
        path, external = self.zip_fixture(members)
        return verifier.verify_artifact(self.root, path, external, self.records, expected_members=3, **kwargs)

    def test_all_members_including_manifest_pass(self):
        self.assertEqual(self.verify(), 3)

    def test_published_count_is_not_relaxed_by_default(self):
        path, external = self.zip_fixture()
        with self.assertRaisesRegex(verifier.VerificationError, "declared archive member count"):
            verifier.verify_artifact(self.root, path, external, self.records)

    def test_archive_whole_size_and_hash(self):
        path, external = self.zip_fixture()
        for changed in ({**external, "bytes": external["bytes"] + 1}, {**external, "sha256": "0" * 64}):
            with self.subTest(changed=changed), self.assertRaisesRegex(verifier.VerificationError, "mismatch"):
                verifier.verify_artifact(self.root, path, changed, self.records, expected_members=3)

    def test_member_payload_tamper_with_updated_outer_zip_digest(self):
        members = list(self.payloads.items()) + [(verifier.DELIVERY_MANIFEST, (self.root / verifier.DELIVERY_MANIFEST).read_bytes())]
        members[0] = (members[0][0], b"X" * len(members[0][1]))
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            self.verify(members)

    def test_manifest_itself_is_compared_to_repository_bytes(self):
        members = list(self.payloads.items()) + [(verifier.DELIVERY_MANIFEST,
                                                b"X" * len((self.root / verifier.DELIVERY_MANIFEST).read_bytes()))]
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            self.verify(members)

    def test_stale_repository_delivery_manifest_binding(self):
        path, external = self.zip_fixture()
        self.write_json(verifier.DELIVERY_MANIFEST, {"records": [record("different", b"anything")]})
        with self.assertRaisesRegex(verifier.VerificationError, "mismatch"):
            verifier.verify_artifact(self.root, path, external, self.records, expected_members=3)

    def test_duplicate_zip_member(self):
        members = [("source.py", self.payloads["source.py"])] * 2 + [
            (verifier.DELIVERY_MANIFEST, (self.root / verifier.DELIVERY_MANIFEST).read_bytes())]
        with self.assertRaisesRegex(verifier.VerificationError, "Duplicate ZIP"):
            self.verify(members)

    def test_missing_zip_member(self):
        with self.assertRaisesRegex(verifier.VerificationError, "ZIP member count"):
            self.verify(list(self.payloads.items()))

    def test_extra_zip_member(self):
        members = list(self.payloads.items()) + [(verifier.DELIVERY_MANIFEST,
                                                (self.root / verifier.DELIVERY_MANIFEST).read_bytes()), ("extra", b"extra")]
        with self.assertRaisesRegex(verifier.VerificationError, "ZIP member count"):
            self.verify(members)

    def test_unsafe_zip_paths(self):
        for name in ("../escape", "/absolute", "nested//file", "nested\\file", "./file"):
            members = [(name, self.payloads["source.py"]), ("snapshots/state.pt", self.payloads["snapshots/state.pt"]),
                       (verifier.DELIVERY_MANIFEST, (self.root / verifier.DELIVERY_MANIFEST).read_bytes())]
            with self.subTest(name=name), self.assertRaises(verifier.VerificationError):
                self.verify(members)

    def test_symlink_zip_entry(self):
        info = zipfile.ZipInfo("source.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        members = [(info, self.payloads["source.py"]), ("snapshots/state.pt", self.payloads["snapshots/state.pt"]),
                   (verifier.DELIVERY_MANIFEST, (self.root / verifier.DELIVERY_MANIFEST).read_bytes())]
        with self.assertRaisesRegex(verifier.VerificationError, "Non-regular ZIP"):
            self.verify(members)

    def test_symlink_artifact(self):
        path, external = self.zip_fixture()
        alias = self.root / "alias.zip"
        alias.symlink_to(path.name)
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            verifier.verify_artifact(self.root, alias, external, self.records, expected_members=3)


class CliTests(Fixtures):
    def test_no_artifact_warns_instead_of_claiming_tensor_validation(self):
        output = io.StringIO()
        with mock.patch.object(verifier, "verify_phase1", return_value=128), \
                mock.patch.object(verifier, "verify_phase2", return_value=({}, verifier.EXTERNAL_ARTIFACT)), \
                contextlib.redirect_stdout(output):
            self.assertEqual(verifier.main([], root=self.root), 0)
        self.assertIn("NOT PROVIDED", output.getvalue())
        self.assertIn("snapshot/tensor files were NOT CHECKED", output.getvalue())

    def test_failure_returns_nonzero(self):
        errors = io.StringIO()
        with mock.patch.object(verifier, "verify_phase1", side_effect=verifier.VerificationError("fixture failure")), \
                contextlib.redirect_stderr(errors):
            self.assertEqual(verifier.main([], root=self.root), 1)
        self.assertIn("FAILED: fixture failure", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
