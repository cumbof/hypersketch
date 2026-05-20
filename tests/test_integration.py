"""
Hypermash integration tests.

These tests exercise the compiled ``hypermash`` binary end-to-end through its
CLI, covering all commands and key error-handling paths.  They are intentionally
written to be familiar to anyone who knows Python's standard ``pytest``.

Run with:
    make test-integration        # from the repository root
    cd tests && pytest -v        # directly with pytest
"""

import os
import re
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY    = os.path.join(REPO_ROOT, "hypermash")
FIXTURES  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

SIMPLE_FASTA     = os.path.join(FIXTURES, "simple.fna")
DIFFERENT_FASTA  = os.path.join(FIXTURES, "different.fna")
AMBIGUOUS_FASTA  = os.path.join(FIXTURES, "ambiguous.fna")
MULTICONTIG_FASTA = os.path.join(FIXTURES, "multi_contig.fna")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(*args, **kwargs):
    """Run the hypermash binary and return a CompletedProcess."""
    return subprocess.run(
        [BINARY, *args],
        capture_output=True,
        text=True,
        **kwargs,
    )


def mash_dist_from(output: str) -> float:
    """Extract the Mash distance value from two-file ``dist`` output."""
    for line in output.splitlines():
        if "Mash Distance" in line:
            return float(line.split(":")[-1].strip())
    raise ValueError(f"Mash distance not found in output:\n{output}")


# ---------------------------------------------------------------------------
# Session-scoped fixture: build the binary once before any test runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def built_binary():
    result = subprocess.run(
        ["g++", "-O3", "-std=c++17", "-pthread", "-march=native",
         "hypermash.cpp", "-o", "hypermash"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build hypermash:\n{result.stderr}")


# ---------------------------------------------------------------------------
# 1. Basic commands
# ---------------------------------------------------------------------------

class TestBasicCommands:
    def test_version_exits_zero_and_prints_version(self):
        r = run("version")
        assert r.returncode == 0
        assert "1.0" in r.stdout

    def test_help_exits_zero_and_lists_commands(self):
        r = run("help")
        assert r.returncode == 0
        for cmd in ("sketch", "dist", "info", "recall"):
            assert cmd in r.stdout

    def test_help_flag_works(self):
        assert run("--help").returncode == 0

    def test_no_args_exits_non_zero(self):
        # No arguments → usage error
        assert run().returncode != 0

    def test_unknown_command_exits_non_zero(self):
        assert run("unknown_command").returncode != 0


# ---------------------------------------------------------------------------
# 2. sketch command
# ---------------------------------------------------------------------------

class TestSketchCommand:
    def test_sketch_creates_hms_file(self, tmp_path):
        out = str(tmp_path / "out")
        r = run("sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, SIMPLE_FASTA)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(out + ".hms")

    def test_sketch_hms_starts_with_magic_bytes(self, tmp_path):
        out = str(tmp_path / "out")
        run("sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, SIMPLE_FASTA)
        import struct
        with open(out + ".hms", "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]  # little-endian uint32
        assert magic == 0x484D5348  # "HMSH" as a 32-bit integer

    def test_sketch_default_output_name(self, tmp_path):
        dst = str(tmp_path / "simple.fna")
        shutil.copy(SIMPLE_FASTA, dst)
        r = run("sketch", "-k", "4", "-d", "256", "-m", "64", dst)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(str(tmp_path / "simple.hms"))

    def test_sketch_ambiguous_bases_do_not_crash(self, tmp_path):
        out = str(tmp_path / "out")
        r = run("sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, AMBIGUOUS_FASTA)
        assert r.returncode == 0, r.stderr

    def test_sketch_multi_contig_fasta(self, tmp_path):
        out = str(tmp_path / "out")
        r = run("sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, MULTICONTIG_FASTA)
        assert r.returncode == 0, r.stderr

    def test_sketch_missing_fasta_exits_non_zero(self):
        assert run("sketch", "/nonexistent/file.fna").returncode != 0

    def test_sketch_k_above_32_exits_non_zero(self, tmp_path):
        out = str(tmp_path / "out")
        r = run("sketch", "-k", "33", "-d", "256", "-m", "64", "-o", out, SIMPLE_FASTA)
        assert r.returncode != 0

    def test_sketch_d_not_multiple_of_8_exits_non_zero(self, tmp_path):
        out = str(tmp_path / "out")
        r = run("sketch", "-k", "4", "-d", "100", "-m", "64", "-o", out, SIMPLE_FASTA)
        assert r.returncode != 0

    def test_sketch_custom_parameters_are_stored(self, tmp_path):
        out = str(tmp_path / "out")
        run("sketch", "-k", "7", "-d", "512", "-m", "128", "-o", out, SIMPLE_FASTA)
        r = run("info", out + ".hms")
        assert "7"   in r.stdout  # k
        assert "512" in r.stdout  # D
        assert "128" in r.stdout  # M


# ---------------------------------------------------------------------------
# 3. dist command
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def two_sketches(tmp_path_factory):
    """Build one sketch for 'simple' and one for 'different', class-scoped."""
    d = tmp_path_factory.mktemp("sketches")
    def sketch(name, fasta):
        out = str(d / name)
        subprocess.run(
            [BINARY, "sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, fasta],
            check=True, capture_output=True,
        )
        return out + ".hms"
    return sketch("simple", SIMPLE_FASTA), sketch("different", DIFFERENT_FASTA)


class TestDistCommand:
    def test_self_distance_is_zero(self, two_sketches):
        s1, _ = two_sketches
        r = run("dist", s1, s1)
        assert r.returncode == 0, r.stderr
        assert mash_dist_from(r.stdout) == pytest.approx(0.0, abs=1e-9)

    def test_dist_output_contains_three_metrics(self, two_sketches):
        s1, _ = two_sketches
        r = run("dist", s1, s1)
        assert "Raw HDC Similarity"  in r.stdout
        assert "Estimated Jaccard"   in r.stdout
        assert "Est. Mash Distance"  in r.stdout

    def test_different_sequences_have_nonzero_distance(self, two_sketches):
        s1, s2 = two_sketches
        r = run("dist", s1, s2)
        assert r.returncode == 0, r.stderr
        assert mash_dist_from(r.stdout) > 0.0

    def test_self_distance_less_than_cross_distance(self, two_sketches):
        s1, s2 = two_sketches
        d_self  = mash_dist_from(run("dist", s1, s1).stdout)
        d_cross = mash_dist_from(run("dist", s1, s2).stdout)
        assert d_self < d_cross

    def test_dist_missing_sketch_exits_non_zero(self, two_sketches):
        s1, _ = two_sketches
        assert run("dist", s1, "/nonexistent/path.hms").returncode != 0

    def test_dist_single_file_exits_non_zero(self, two_sketches):
        s1, _ = two_sketches
        assert run("dist", s1).returncode != 0

    def test_dist_list_mode_produces_matrix_header(self, two_sketches, tmp_path):
        # Matrix mode (with header) requires 3+ files; 2 files use the two-file direct path.
        s1, s2 = two_sketches
        lst = tmp_path / "files.txt"
        lst.write_text(f"{s1}\n{s1}\n{s2}\n")  # 3 sketches → matrix mode
        r = run("dist", "-l", str(lst))
        assert r.returncode == 0, r.stderr
        assert "genome_1" in r.stdout and "genome_2" in r.stdout

    def test_dist_list_mode_n_lines_equals_n_pairs_plus_header(self, two_sketches, tmp_path):
        s1, s2 = two_sketches
        lst = tmp_path / "files.txt"
        lst.write_text(f"{s1}\n{s1}\n{s2}\n")   # 3 sketches → 3 pairs
        r = run("dist", "-l", str(lst))
        assert r.returncode == 0, r.stderr
        data_lines = [l for l in r.stdout.splitlines() if l and "genome_1" not in l]
        assert len(data_lines) == 3


# ---------------------------------------------------------------------------
# 4. info command
# ---------------------------------------------------------------------------

class TestInfoCommand:
    @pytest.fixture
    def sketch_file(self, tmp_path):
        out = str(tmp_path / "out")
        subprocess.run(
            [BINARY, "sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, SIMPLE_FASTA],
            check=True, capture_output=True,
        )
        return out + ".hms"

    def test_info_exits_zero(self, sketch_file):
        assert run("info", sketch_file).returncode == 0

    def test_info_shows_k_d_m(self, sketch_file):
        r = run("info", sketch_file)
        assert "k-mer size"      in r.stdout
        assert "Dimensions"      in r.stdout
        assert "Memory Bank Size" in r.stdout

    def test_info_shows_correct_parameter_values(self, sketch_file):
        r = run("info", sketch_file)
        # Values 4, 256, 64 must all appear in the output
        assert re.search(r"\b4\b",   r.stdout)
        assert re.search(r"\b256\b", r.stdout)
        assert re.search(r"\b64\b",  r.stdout)

    def test_info_missing_file_exits_non_zero(self):
        assert run("info", "/nonexistent/path.hms").returncode != 0

    def test_info_invalid_file_exits_non_zero(self, tmp_path):
        bad = tmp_path / "bad.hms"
        bad.write_bytes(b"not a valid hms file at all")
        assert run("info", str(bad)).returncode != 0


# ---------------------------------------------------------------------------
# 5. recall command
# ---------------------------------------------------------------------------

class TestRecallCommand:
    @pytest.fixture
    def sketch_and_fasta(self, tmp_path):
        out = str(tmp_path / "out")
        subprocess.run(
            [BINARY, "sketch", "-k", "4", "-d", "256", "-m", "64", "-o", out, SIMPLE_FASTA],
            check=True, capture_output=True,
        )
        return out + ".hms", SIMPLE_FASTA

    def test_recall_exits_zero(self, sketch_and_fasta):
        hms, fna = sketch_and_fasta
        assert run("recall", hms, fna).returncode == 0

    def test_recall_output_is_tsv_with_four_columns(self, sketch_and_fasta):
        hms, fna = sketch_and_fasta
        r = run("recall", hms, fna)
        assert r.returncode == 0, r.stderr
        cols = r.stdout.strip().split("\t")
        assert len(cols) == 4

    def test_recall_value_is_between_0_and_1(self, sketch_and_fasta):
        hms, fna = sketch_and_fasta
        r = run("recall", hms, fna)
        recall_val = float(r.stdout.strip().split("\t")[2])
        assert 0.0 <= recall_val <= 1.0

    def test_recall_missing_sketch_exits_non_zero(self):
        assert run("recall", "/nonexistent.hms", SIMPLE_FASTA).returncode != 0
