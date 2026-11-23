"""A placement that is stopped while it cleans up names what it leaves behind, as the names are hidden (#1056)."""

import errno
import os
import shutil
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, tracker_dirs

from salmon import placement
from salmon.placement import PlacedFolders, PlacementError


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album")
    album(found.red / "Album", tags={"album": "the one there"})
    answer(monkeypatch, True)
    return found


def _interrupted(prefix: str) -> Callable[..., None]:
    rmtree = shutil.rmtree

    def remove(path: str, *args: Any, **kwargs: Any) -> None:
        if prefix in os.path.basename(path):
            raise KeyboardInterrupt
        rmtree(path, *args, **kwargs)

    return remove


@pytest.mark.parametrize("prefix", [placement.REPLACED_PREFIX, placement.BUILDING_PREFIX], ids=["replaced", "building"])
def test_an_interrupted_cleanup_names_what_is_left(monkeypatch, dirs, capsys, prefix: str) -> None:
    monkeypatch.setattr(shutil, "rmtree", _interrupted(prefix))

    with pytest.raises(KeyboardInterrupt):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    [left] = dirs.red.glob(f"{prefix}*")
    assert str(left) in capsys.readouterr().out


def test_a_folder_that_cannot_be_looked_at_is_not_taken_for_removed(monkeypatch, dirs, capsys) -> None:
    real = os.stat

    def cannot_look(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if isinstance(path, str) and os.path.basename(path).startswith(placement.REPLACED_PREFIX):
            raise OSError(errno.EIO, "I/O error", path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", cannot_look)

    anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    if left := list(dirs.red.glob(f"{placement.REPLACED_PREFIX}*")):
        assert str(left[0]) in capsys.readouterr().out


def test_an_interrupted_undo_names_the_folder_moved_aside(monkeypatch, dirs, capsys) -> None:
    rename = os.rename

    def interrupted_twice(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if placement.REPLACED_PREFIX in str(src):
            raise KeyboardInterrupt
        if placement.BUILDING_PREFIX in str(src):
            raise OSError(errno.EIO, "I/O error")
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", interrupted_twice)

    with pytest.raises((KeyboardInterrupt, PlacementError)):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    [left] = dirs.red.glob(f"{placement.REPLACED_PREFIX}*")
    assert str(left) in capsys.readouterr().out
