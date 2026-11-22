"""Whatever stops a placement, no folder is left with another's conversion record (#1056).

A FLAC with a transcode's record is uploaded with a false description, so each way of stopping has a case.
"""

import errno
import json
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, contents, tracker_dirs

from salmon import placement
from salmon.converter import conversions
from salmon.converter.conversions import conversion_of, record_conversion
from salmon.placement import PlacedFolders, PlacementError

RECORDS = ["none", "source", "old", "both"]


def _world(monkeypatch, root: Path, records: str, *, replaces: bool = True) -> SimpleNamespace:
    """An album to place and, if it replaces, the folder there, with the records named; "old" is stale with none."""
    found = tracker_dirs(monkeypatch, root)
    found.album = album(found.downloads / "Album")
    found.dest = found.red / "Album"
    found.new, found.old = contents(found.album), None
    found.red.mkdir(parents=True)
    if replaces:
        found.old = contents(album(found.dest, tags={"album": "the one there"}))
    if records in ("source", "both"):
        record_conversion(str(found.album), source="/new", kind="transcode", bitrate="V0")
    if records in ("old", "both"):
        record_conversion(str(found.dest), source="/old", kind="transcode", bitrate="320")
    found.new_source = "/new" if records in ("source", "both") else None
    found.old_source = "/old" if records in ("old", "both") else None
    answer(monkeypatch, True)
    return found


def _place(world: SimpleNamespace) -> None:
    anyio.run(PlacedFolders().place, str(world.album), "RED")


def _recorded(dest: Path) -> str | None:
    """The source in the record at dest's name, whether or not a folder is there."""
    try:
        return json.loads((dest.parent / ".salmon-conversions" / f"{dest.name}.json").read_text())["source"]
    except FileNotFoundError:
        return None


def _stopping_at(real: Callable[..., Any], failing: int, calls: list[int], how: str) -> Callable[..., Any]:
    """`real`, whose call number `failing` fails, or is interrupted as it returns."""

    def call(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) != failing:
            return real(*args, **kwargs)
        if how == "error":
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real(*args, **kwargs)
        raise KeyboardInterrupt

    return call


@pytest.mark.parametrize("replaces", [True, False], ids=["replacing", "new"])
@pytest.mark.parametrize("records", RECORDS)
@pytest.mark.parametrize("how", ["error", "Ctrl-C"])
@pytest.mark.parametrize("call", ["rename", "replace", "remove", "lstat"], ids=lambda call: f"os.{call}")
def test_whichever_call_stops_it_each_folder_has_its_own_record(
    monkeypatch, tmp_path, call: str, how: str, records: str, replaces: bool
) -> None:
    real = getattr(os, call)
    failing = 0
    while True:  # Each of this call that a placement makes stops a run of its own
        failing += 1
        root = tmp_path / str(failing)
        root.mkdir()
        world = _world(monkeypatch, root, records, replaces=replaces)
        calls: list[int] = []

        with monkeypatch.context() as faulty:
            faulty.setattr(os, call, _stopping_at(real, failing, calls, how))
            with suppress(PlacementError, OSError, KeyboardInterrupt):
                _place(world)

        there = contents(world.dest) if world.dest.exists() else None
        assert there in (world.old, world.new), f"call {failing}"
        expected = world.new_source if there == world.new else world.old_source
        assert _recorded(world.dest) == expected, f"call {failing}"
        if len(calls) < failing:
            return


@pytest.mark.parametrize("records", RECORDS)
@pytest.mark.parametrize("killed", ["set aside", "moved aside", "moved in", "record written"])
def test_a_kill_leaves_no_folder_with_anothers_record(monkeypatch, tmp_path, killed: str, records: str) -> None:
    world = _world(monkeypatch, tmp_path, records)
    rename, replace, copy_record = os.rename, os.replace, conversions.copy_record

    def killed_after_rename(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        rename(src, dst, *args, **kwargs)
        if killed == "moved aside" and placement.REPLACED_PREFIX in str(dst):
            raise KeyboardInterrupt
        if killed == "moved in" and placement.BUILDING_PREFIX in str(src):
            raise KeyboardInterrupt

    def killed_after_replace(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        replace(src, dst, *args, **kwargs)
        if killed == "set aside" and placement.REPLACED_PREFIX in str(dst):
            raise KeyboardInterrupt

    def killed_after_record(src: str, dst: str) -> None:
        copy_record(src, dst)
        if killed == "record written":
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "rename", killed_after_rename)
    monkeypatch.setattr(os, "replace", killed_after_replace)
    monkeypatch.setattr(conversions, "copy_record", killed_after_record)
    monkeypatch.setattr(placement, "_undo", lambda *_args: None)  # A kill runs none of it

    with suppress(KeyboardInterrupt):
        _place(world)

    if not world.dest.exists():
        # The user puts the old one back by hand: its record may be gone, never another's
        assert _recorded(world.dest) in (None, world.old_source)
        return
    own = world.new_source if contents(world.dest) == world.new else world.old_source
    assert _recorded(world.dest) in (None, own)


@pytest.mark.parametrize("stuck", ["the new folder", "the old folder"])
def test_an_undo_that_cannot_finish_leaves_no_false_record(monkeypatch, tmp_path, stuck: str) -> None:
    world = _world(monkeypatch, tmp_path, "both")
    rename, replace = os.rename, os.replace
    sidecar = str(world.red / ".salmon-conversions" / "Album.json")

    def interrupted_at_the_record(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        replace(src, dst, *args, **kwargs)
        if str(dst) == sidecar:
            raise KeyboardInterrupt

    def cannot_undo(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        out, back = str(src) == str(world.dest), placement.REPLACED_PREFIX in str(src)
        into_building = out and placement.BUILDING_PREFIX in str(dst)
        if (stuck == "the new folder" and into_building) or (stuck == "the old folder" and back):
            raise OSError(errno.EIO, "I/O error")
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", interrupted_at_the_record)
    monkeypatch.setattr(os, "rename", cannot_undo)
    with pytest.raises(KeyboardInterrupt):
        _place(world)

    if stuck == "the new folder":
        assert _recorded(world.dest) in (None, "/new")
        world.dest.rename(world.red / "taken away")  # As the user would, to put the old one back
    [kept] = world.red.glob(".salmon-replaced-*")
    rename(kept, world.dest)  # As salmon told them

    assert contents(world.dest) == world.old
    assert _recorded(world.dest) in (None, "/old")


def test_a_folder_that_appears_takes_no_record_of_the_one_moved_aside(monkeypatch, tmp_path) -> None:
    world = _world(monkeypatch, tmp_path, "old")
    rename = os.rename

    def then_one_appears(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        rename(src, dst, *args, **kwargs)
        if str(src) == str(world.dest):
            album(world.dest, tags={"album": "a third"})

    monkeypatch.setattr(os, "rename", then_one_appears)

    with pytest.raises(PlacementError, match="appeared"):
        _place(world)

    assert conversion_of(str(world.dest)) is None  # A FLAC, never described as a transcode


@pytest.mark.parametrize("replaces", [True, False], ids=["replacing", "new"])
def test_a_placement_done_leaves_no_record_set_aside(monkeypatch, tmp_path, replaces: bool) -> None:
    world = _world(monkeypatch, tmp_path, "both", replaces=replaces)

    _place(world)

    assert [file.name for file in (world.red / ".salmon-conversions").iterdir()] == ["Album.json"]
    assert _recorded(world.dest) == "/new"
