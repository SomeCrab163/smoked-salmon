"""salmon up with per-tracker directories (#1056): the first tracker's folder, the later ones', and the conversions'.

They guard deletion of users' data, so each guard has a case of its own.
"""

import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import asyncclick as click
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, building_left, contents, inodes, links_fail, tracker_dirs
from helpers.uploader_dry_run import METADATA, RENAMED, _album, _returning, _run_up
from test_uploader_dry_run import (
    image_uploads,  # noqa: F401 (a fixture: see images)
)

import salmon.tagger.foldername
from salmon import cfg, interaction
from salmon.converter.conversions import conversion_of, record_conversion
from salmon.errors import UploadError
from salmon.tagger.foldername import rename_folder
from salmon.uploader import record

# Covers and spectrals go to the fake image host.
pytestmark = pytest.mark.usefixtures("images")


@pytest.fixture
def images(image_uploads) -> list[tuple[str, str]]:  # noqa: F811
    return image_uploads


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.torrents = tmp_path / "torrents"
    found.torrents.mkdir()
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", False)
    return found


def _up(monkeypatch, dirs: SimpleNamespace, source: Path, *trackers: str, conversions: str = "none", **kwargs: Any):
    """salmon up -t `trackers`, with what it would exit with."""
    with record.recording() as recorded:
        run = _run_up(
            monkeypatch,
            source,
            dirs.torrents,
            ("--conversions", conversions, *kwargs.pop("args", ())),
            trackers=trackers,
            **kwargs,
        )
    return run, recorded.exit_status(run.result.exit_code)


def _rename(monkeypatch, path: Path, name: str, parent: Path | None, **kwargs: Any) -> str:
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", _returning(name))
    given = str(parent) if parent else None
    return anyio.run(partial(rename_folder, str(path), dict(METADATA), True, parent=given, **kwargs))


def _earlier(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "other.flac").write_bytes(b"another album")
    os.utime(folder, ns=(1_000_000_000, 1_000_000_000))
    return folder


# --- The first tracker's directory (T17) -----------------------------------------------------------------------------


def test_up_renames_into_the_first_trackers_directory_and_makes_it(monkeypatch, dirs) -> None:
    source = _album(dirs.root / "in" / "Album")

    run, status = _up(monkeypatch, dirs, source, "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert f"Created {dirs.red}." in run.result.output
    assert f"Hardlinked folder to '{dirs.red / RENAMED}'." in run.result.output
    assert len(list((dirs.red / RENAMED).glob("*.flac"))) == 2
    assert not (dirs.downloads / RENAMED).exists()


def test_a_first_tracker_without_a_key_gets_the_global_directory(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert (dirs.downloads / RENAMED).is_dir()


def test_with_no_key_up_renames_into_the_global_one_and_copies_silently(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)
    links_fail(monkeypatch)
    asked = answer(monkeypatch)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "Hardlinking didn't work, falling back to non-hardlink copy..." in run.result.output
    assert (dirs.downloads / RENAMED).is_dir()
    assert asked == []


def test_a_rename_that_cannot_link_under_yyy_copies_and_says_so(monkeypatch, dirs) -> None:
    links_fail(monkeypatch)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "-yyy: copying" in run.result.output
    assert building_left(dirs.root) == []


def test_a_declined_copy_aborts_the_rename_and_keeps_the_folder_to_replace(monkeypatch, dirs) -> None:
    source = album(dirs.root / "in" / "Album")
    before = contents(_earlier(dirs.red / "New"))
    links_fail(monkeypatch)
    answer(monkeypatch, True, False)  # Replace it; don't copy

    with pytest.raises(click.Abort):
        _rename(monkeypatch, source, "New", dirs.red, check=True)

    assert contents(dirs.red / "New") == before
    assert len(list(source.glob("*.flac"))) == 2


def test_the_renamed_folder_replaces_whole_and_never_merges(monkeypatch, dirs) -> None:
    source = album(dirs.root / "in" / "New")
    record_conversion(str(_earlier(dirs.red / "New")), source="/of/the/old", kind="transcode", bitrate="V0")
    answer(monkeypatch, True)

    assert _rename(monkeypatch, source, "New", dirs.red) == str(dirs.red / "New")

    assert contents(dirs.red / "New") == contents(source)
    assert conversion_of(str(dirs.red / "New")) is None


def test_keeping_the_folder_there_stops_the_rename(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs.red / "New"))
    asked = answer(monkeypatch, False)

    with pytest.raises(UploadError, match="already exists"):
        _rename(monkeypatch, album(dirs.root / "in" / "New"), "New", dirs.red)

    assert asked[0][1]["default"] is False
    assert contents(dirs.red / "New") == before


def test_a_folder_recreated_during_the_renames_question_is_not_replaced(monkeypatch, dirs) -> None:
    dest = _earlier(dirs.red / "New")

    async def recreate_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        (dest / "other.flac").unlink()
        dest.rmdir()
        dest.mkdir()
        (dest / "new download.flac").write_bytes(b"new")
        return True

    monkeypatch.setattr(interaction, "confirm", recreate_then_agree)

    with pytest.raises(click.Abort):
        _rename(monkeypatch, album(dirs.root / "in" / "New"), "New", dirs.red)

    assert [file.name for file in dest.iterdir()] == ["new download.flac"]


def test_a_run_under_yyy_aborts_at_a_folder_already_there(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs.red / RENAMED))

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert status == 1, run.result.output
    assert "Not replacing" in run.result.output and "Aborting upload" in run.result.output
    assert "would you like to replace it?" not in run.result.output
    assert contents(dirs.red / RENAMED) == before
    assert run.tracker.not_gets() == []


@pytest.mark.parametrize("which", ["inside the album", "holding the album", "holding the original", "a link"])
def test_a_rename_never_goes_over_what_the_run_works_from(monkeypatch, dirs, which) -> None:
    original, parent = None, dirs.red
    if which == "inside the album":
        source = album(dirs.root / "box")
        parent = source / "red"
        parent.mkdir()
        monkeypatch.setattr(cfg.tracker.red, "download_directory", str(parent))
    elif which == "holding the album":
        source = album(dirs.red / "New" / "CD1")
    elif which == "holding the original":
        original = album(dirs.red / "New" / "CD1")
        source = album(dirs.root / "copy")
    else:
        source = album(dirs.root / "in")
        album(dirs.red / "real")
        (dirs.red / "New").symlink_to(dirs.red / "real", target_is_directory=True)
    before = contents(dirs.root)
    asked = answer(monkeypatch)

    with pytest.raises(click.Abort):
        _rename(monkeypatch, source, "New", parent, original=str(original) if original else None)

    assert asked == []
    assert contents(dirs.root) == before


# --- A data directory is never replaced (T18) ------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["up", "tag"])
@pytest.mark.parametrize("there", ["seeding", "not made yet", "holding it"])
def test_an_album_named_like_a_trackers_root_never_replaces_it(monkeypatch, dirs, command, there) -> None:
    tracker_root = dirs.downloads / "red" / ("ops" if there == "holding it" else "")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(tracker_root))
    if there != "not made yet":
        album(tracker_root / "Seeding")
    source = album(dirs.root / "in" / "Album")
    before = contents(dirs.downloads)
    asked = answer(monkeypatch, True, True)

    # salmon up gives the global directory as the parent of a first tracker without a key; salmon tag gives none.
    with pytest.raises(UploadError, match="which is a download_directory"):
        _rename(monkeypatch, source, "red", dirs.downloads if command == "up" else None)

    assert asked == []
    assert contents(dirs.downloads) == before


def test_salmon_tags_rename_is_otherwise_unchanged(monkeypatch, dirs, capsys) -> None:
    links_fail(monkeypatch)
    asked = answer(monkeypatch)

    made = _rename(monkeypatch, album(dirs.root / "in" / "Album"), "New", None)

    assert made == str(dirs.downloads / "New")
    assert "falling back to non-hardlink copy" in capsys.readouterr().out
    assert asked == []


# --- An album worked on as a copy (T19) ------------------------------------------------------------------------------


@pytest.mark.parametrize("why", ["library", "shares files"])
def test_a_copy_ends_in_the_trackers_directory_and_the_original_is_untouched(monkeypatch, dirs, why) -> None:
    if why == "library":
        monkeypatch.setattr(cfg.directory, "library_dirs", [str(dirs.library)])
        source = _album(dirs.library / "Album")
    else:
        source = _album(dirs.root / "in" / "Album")
        os.link(source / "cover.jpg", dirs.root / "seeded.jpg")
    before = contents(source), inodes(source)

    run, status = _up(monkeypatch, dirs, source, "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "The renamed copy stays where it is renamed to." in run.result.output
    assert (contents(source), inodes(source)) == before
    assert len(list((dirs.red / RENAMED).glob("*.flac"))) == 2
    assert not set(inodes(dirs.red / RENAMED).values()) & set(inodes(source).values())


def test_a_staged_copy_on_another_mount_asks_before_it_is_copied(monkeypatch, dirs) -> None:
    staged = album(dirs.downloads / ".salmon-staging" / "run-1" / "Album")
    links_fail(monkeypatch)
    asked = answer(monkeypatch, True)

    _rename(monkeypatch, staged, "New", dirs.red)

    assert "Copy instead" in asked[0][0]
