"""remove_source_dir keeps an album that a folder a tracker was skipped over needs (#1056).

They guard deletion of users' data.
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, tracker_dirs

from salmon import cfg, interaction
from salmon.placement import PlacedFolders, PlacementDeclined, PlacementError

Arrange = Callable[[SimpleNamespace], None]


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.given = album(found.root / "in" / "Album")
    (found.given / "Scans").mkdir()
    (found.given / "Scans" / "back.jpg").write_bytes(b"back")
    # Renamed into the first tracker's directory, and placed from there
    found.renamed = album(found.red / "New")
    found.ops.mkdir(parents=True)
    found.dest = found.ops / "New"
    return found


def _to_the_album(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(dirs.given, target_is_directory=True)


def _to_a_subfolder(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(dirs.given / "Scans", target_is_directory=True)


def _to_what_holds_it(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(dirs.given.parent, target_is_directory=True)


def _relative(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(os.path.relpath(dirs.given / "Scans", dirs.ops), target_is_directory=True)


def _through_another_link(dirs: SimpleNamespace) -> None:
    between = dirs.root / "between"
    between.symlink_to(dirs.given / "Scans", target_is_directory=True)
    dirs.dest.symlink_to(between, target_is_directory=True)


def _to_a_file(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(dirs.given / "01.flac")


def _folder_of_file_links(dirs: SimpleNamespace) -> None:
    dirs.dest.mkdir()
    for track in dirs.given.glob("*.flac"):
        (dirs.dest / track.name).symlink_to(track)


def _folder_link_further_down(dirs: SimpleNamespace) -> None:
    (dirs.dest / "CD1").mkdir(parents=True)
    (dirs.dest / "CD1" / "Scans").symlink_to(dirs.given / "Scans", target_is_directory=True)


NEEDING_IT: list[Arrange] = [
    _to_the_album,
    _to_a_subfolder,
    _to_what_holds_it,
    _relative,
    _through_another_link,
    _to_a_file,
    _folder_of_file_links,
    _folder_link_further_down,
]


def _skipped(dirs: SimpleNamespace) -> str | None:
    placed = PlacedFolders(str(dirs.renamed), str(dirs.given))
    with pytest.raises(PlacementError):
        anyio.run(placed.place, str(dirs.renamed), "OPS")
    return placed.skipped_over(str(dirs.given))


@pytest.mark.parametrize("arrange", NEEDING_IT)
def test_a_skipped_folder_that_links_into_the_album_needs_it(monkeypatch, dirs, arrange: Arrange) -> None:
    arrange(dirs)
    answer(monkeypatch, False)  # Keep the folder there, where it is asked

    assert _skipped(dirs) == str(dirs.dest)


def test_a_folder_kept_under_yyy_needs_it_too(dirs) -> None:
    _folder_of_file_links(dirs)

    with interaction.assuming_defaults():
        assert _skipped(dirs) == str(dirs.dest)


def test_a_folder_whose_replace_failed_still_needs_it(monkeypatch, dirs) -> None:
    _folder_of_file_links(dirs)
    monkeypatch.setattr(cfg.directory, "hardlinks", False)
    answer(monkeypatch, True)

    def full(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", full)

    assert _skipped(dirs) == str(dirs.dest)  # It is still there, with its links


def _linked_elsewhere(dirs: SimpleNamespace) -> None:
    dirs.dest.symlink_to(album(dirs.root / "in" / "Album 2"), target_is_directory=True)


def _holds_a_link_elsewhere(dirs: SimpleNamespace) -> None:
    dirs.dest.mkdir()
    (dirs.dest / "more").symlink_to(album(dirs.root / "elsewhere"), target_is_directory=True)


def _hardlinks_of_it(dirs: SimpleNamespace) -> None:
    shutil.copytree(dirs.given, dirs.dest, copy_function=os.link)  # They outlive it


@pytest.mark.parametrize("arrange", [_linked_elsewhere, _holds_a_link_elsewhere, _hardlinks_of_it])
def test_a_skipped_folder_that_does_not_need_the_album_keeps_nothing(monkeypatch, dirs, arrange: Arrange) -> None:
    arrange(dirs)
    answer(monkeypatch, False)

    assert _skipped(dirs) is None


def test_a_folder_that_was_replaced_needs_nothing(monkeypatch, dirs) -> None:
    _folder_of_file_links(dirs)
    answer(monkeypatch, True)
    placed = PlacedFolders(str(dirs.renamed), str(dirs.given))

    made = anyio.run(placed.place, str(dirs.renamed), "OPS")

    assert Path(made) == dirs.dest
    assert placed.skipped_over(str(dirs.given)) is None


# --- A data directory a tracker was skipped over ---------------------------------------------------------------------


def _skipped_over_a_data_directory(monkeypatch, tmp_path: Path) -> tuple[PlacedFolders, SimpleNamespace]:
    """OPS, with no folder of its own, declined the global one's `red`, which is RED's folder of albums."""
    dirs = tracker_dirs(monkeypatch, tmp_path, "RED")
    monkeypatch.setattr(cfg.directory, "download_directory", str(tmp_path / "data"))
    dirs.held = [album(dirs.red / name) for name in ("One", "Two")]
    placed = PlacedFolders()
    with pytest.raises(PlacementDeclined, match="download_directory"):
        anyio.run(placed.place, str(album(tmp_path / "in" / "red")), "OPS")
    return placed, dirs


def test_a_data_directory_a_tracker_was_skipped_over_is_not_walked(monkeypatch, tmp_path) -> None:
    placed, dirs = _skipped_over_a_data_directory(monkeypatch, tmp_path)
    walked: list[str] = []
    walk = os.walk

    def recording(top: Any, *args: Any, **kwargs: Any) -> Any:
        walked.append(os.fspath(top))
        return walk(top, *args, **kwargs)

    monkeypatch.setattr(os, "walk", recording)

    assert placed.skipped_over(str(tmp_path / "in" / "red")) is None
    assert str(dirs.red) not in walked


def test_a_data_directory_a_tracker_was_skipped_over_still_needs_an_album_in_it(monkeypatch, tmp_path) -> None:
    placed, dirs = _skipped_over_a_data_directory(monkeypatch, tmp_path)

    assert placed.skipped_over(str(dirs.held[0])) == str(dirs.red)
