"""Check revision validation with real Git repositories and CMake."""

import shutil
import subprocess
from pathlib import Path

import pytest

VERIFY_REVISION = (
    Path(__file__).resolve().parents[1]
    / "cmake"
    / "VerifyLibdiffpyRevision.cmake"
)
FETCH_LIBDIFFPY = VERIFY_REVISION.with_name("FetchLibdiffpy.cmake")


@pytest.fixture
def fetch_source(tmp_path):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("Build configuration checks require CMake.")
    project = tmp_path / "fetch-project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(FetchCheck LANGUAGES NONE)\n"
        f'include("{FETCH_LIBDIFFPY.as_posix()}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/source-dir.txt"\n'
        '     "${LIBDIFFPY_SOURCE_DIR}")\n'
    )

    def configure(source):
        build = tmp_path / "fetch-build"
        result = subprocess.run(
            [
                cmake,
                "-S",
                str(project),
                "-B",
                str(build),
                f"-DFETCHCONTENT_SOURCE_DIR_LIBDIFFPY={source}",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE",
            ],
            capture_output=True,
            text=True,
        )
        return result, build

    return configure


def test_local_archive_override_without_git(tmp_path, fetch_source):
    source = tmp_path / "libdiffpy"
    template = source / "src" / "diffpy" / "version.tpl"
    template.parent.mkdir(parents=True)
    template.write_text("// libdiffpy version template\n")
    result, build = fetch_source(source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "source-dir.txt").read_text() == source.as_posix()


def test_local_archive_override_requires_sources(tmp_path, fetch_source):
    source = tmp_path / "empty-libdiffpy"
    source.mkdir()
    result, _ = fetch_source(source)
    assert result.returncode != 0
    assert "Missing libdiffpy sources" in result.stderr


@pytest.fixture
def revision_check(tmp_path):
    cmake = shutil.which("cmake")
    git = shutil.which("git")
    if not cmake or not git:
        pytest.skip("Build configuration checks require CMake and Git.")
    repository = tmp_path / "native-source"
    subprocess.run(
        [git, "init", str(repository)], check=True, capture_output=True
    )
    subprocess.run(
        [
            git,
            "-C",
            str(repository),
            "-c",
            "user.name=Build config test",
            "-c",
            "user.email=build-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "Native source revision",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        [git, "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(RevisionCheck LANGUAGES NONE)\n"
        f'include("{VERIFY_REVISION.as_posix()}")\n'
        'verify_libdiffpy_revision("${NATIVE_SOURCE}"\n'
        '                         "${EXPECTED_REVISION}")\n'
    )

    def configure(source, expected, *options):
        return subprocess.run(
            [
                cmake,
                "-S",
                str(project),
                "-B",
                str(tmp_path / "build"),
                f"-DNATIVE_SOURCE={source}",
                f"-DEXPECTED_REVISION={expected}",
                *options,
            ],
            capture_output=True,
            text=True,
        )

    return repository, revision, configure


def test_matching_git_revision(revision_check):
    repository, revision, configure = revision_check
    # A Git worktree can use a .git file pointing to a separate Git directory.
    git_dir = repository.parent / "native-git-dir"
    (repository / ".git").rename(git_dir)
    (repository / ".git").write_text(f"gitdir: {git_dir.as_posix()}\n")
    result = configure(repository, revision)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mismatched_git_revision(revision_check):
    repository, revision, configure = revision_check
    result = configure(repository, "0" * 40)
    assert result.returncode != 0
    assert "libdiffpy revision mismatch" in result.stderr
    assert revision in result.stderr
    assert "0" * 40 in result.stderr
    assert "FETCHCONTENT_SOURCE_DIR_LIBDIFFPY" in result.stderr


def test_archive_does_not_use_parent_git_revision(revision_check):
    repository, _, configure = revision_check
    archive_source = repository / "unpacked-archive" / "libdiffpy"
    archive_source.mkdir(parents=True)
    result = configure(
        archive_source, "0" * 40, "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_broken_git_metadata_is_not_treated_as_archive(revision_check):
    repository, revision, configure = revision_check
    broken_source = repository / "broken-checkout"
    broken_source.mkdir()
    (broken_source / ".git").write_text("gitdir: missing-git-directory\n")
    result = configure(broken_source, revision)
    assert result.returncode != 0
    assert "Cannot determine the libdiffpy revision" in result.stderr
