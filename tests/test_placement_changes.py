"""What changes inside a folder while it is being replaced is not deleted with it (#1056).

They guard deletion of users' data.
"""

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, building_left, contents, tracker_dirs

from salmon import cfg, interaction, placement
from salmon.placement import PlacedFolders, PlacementDeclined


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album")
    found.dest = album(found.red / "Album", tags={"album": "the one there"})
    found.disc = album(found.dest / "CD2")
    time.sleep(0.02)  # So that a change has a later ctime than the folder's making, in a coarse clock's ticks too
    return found


def _place(dirs: SimpleNamespace) -> str:
    return anyio.run(PlacedFolders().place, str(dirs.album), "RED")


def _added(disc: Path) -> None:
    (disc / "late.flac").write_bytes(b"late")


def _added_deeper(disc: Path) -> None:
    (disc / "Scans").mkdir()
    (disc / "Scans" / "late.jpg").write_bytes(b"late")


def _grown(disc: Path) -> None:
    with open(disc / "01.flac", "ab") as track:
        track.write(b"more")


def _edited_keeping_size_and_mtime(disc: Path) -> None:
    track = disc / "01.flac"
    was = track.stat()
    track.write_bytes(b"x" * was.st_size)
    os.utime(track, ns=(was.st_atime_ns, was.st_mtime_ns))  # As cp -p or a tagger that keeps times


def _removed(disc: Path) -> None:
    (disc / "01.flac").unlink()


def _made_private(disc: Path) -> None:
    (disc / "01.flac").chmod(0o600)


CHANGES = [_added, _added_deeper, _grown, _edited_keeping_size_and_mtime, _removed, _made_private]


@pytest.mark.parametrize("change", CHANGES)
@pytest.mark.parametrize("during", ["question", "copy"])
def test_a_change_below_the_top_level_is_not_replaced(
    monkeypatch, dirs, during: str, change: Callable[[Path], None]
) -> None:
    monkeypatch.setattr(cfg.directory, "hardlinks", False)
    changed: dict[str, str] = {}

    def change_once() -> None:
        if not changed:
            change(dirs.disc)
            changed.update(contents(dirs.dest))

    async def agree(*_args: Any, **_kwargs: Any) -> bool:
        if during == "question":
            change_once()
        return True

    copytree = shutil.copytree

    def copy(*args: Any, **kwargs: Any) -> Any:
        if during == "copy":
            change_once()
        return copytree(*args, **kwargs)

    monkeypatch.setattr(interaction, "confirm", agree)
    monkeypatch.setattr(shutil, "copytree", copy)

    with pytest.raises(PlacementDeclined, match="what is in it changed"):
        _place(dirs)

    assert contents(dirs.dest) == changed
    assert building_left(dirs.root) == []


@pytest.mark.parametrize("change", ["chmod of a shared file", "now shared"])
def test_a_change_of_what_is_shared_is_not_replaced(monkeypatch, dirs, change: str) -> None:
    track, seeded = dirs.disc / "01.flac", dirs.root / "seeded.flac"
    if change != "now shared":
        os.link(track, seeded)  # Its ctime says nothing then

    async def change_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        if change == "now shared":
            os.link(track, seeded)
        else:
            track.chmod(0o600)
        return True

    monkeypatch.setattr(interaction, "confirm", change_then_agree)
    reason = "started sharing" if change == "now shared" else "what is in it changed"

    with pytest.raises(PlacementDeclined, match=reason):
        _place(dirs)


def test_a_change_as_it_is_moved_aside_is_kept_and_named(monkeypatch, dirs, capsys) -> None:
    rename = os.rename

    def written_then_moved(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if placement.REPLACED_PREFIX in str(dst):
            _added(dirs.disc)  # Too late to refuse
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", written_then_moved)
    answer(monkeypatch, True)

    _place(dirs)

    [kept] = dirs.red.glob(".salmon-replaced-*")
    assert (kept / "CD2" / "late.flac").read_bytes() == b"late"
    assert (kept / "01.flac").exists()  # All of it
    assert str(kept) in capsys.readouterr().out
    assert contents(dirs.dest) == contents(dirs.album)


@pytest.mark.parametrize("linked", [True, False], ids=["link", "copy"])
def test_what_an_earlier_run_linked_from_the_album_can_still_be_replaced(monkeypatch, dirs, linked: bool) -> None:
    # A retry after a failed upload. Linking the album again changes the ctime of the files there.
    shutil.rmtree(dirs.dest)
    shutil.copytree(dirs.album, dirs.dest, copy_function=os.link)
    monkeypatch.setattr(cfg.directory, "hardlinks", linked)
    answer(monkeypatch, True)
    time.sleep(0.02)

    _place(dirs)

    assert contents(dirs.dest) == contents(dirs.album)
    assert list(dirs.red.glob(".salmon-replaced-*")) == []
