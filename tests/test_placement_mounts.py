"""A folder with a mount point in it is never replaced (#1056): removing it would empty what is mounted there.

Mounting takes a privilege CI hasn't, so the kernel's table of mounts is a file of the test's.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, contents, tracker_dirs

from salmon import interaction, placement, replace_guard
from salmon.placement import PlacedFolders, PlacementDeclined


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album")
    found.dest = album(found.red / "Album", tags={"album": "the one there"})
    found.inner = album(found.dest / "More music")
    found.table = tmp_path / "mountinfo"
    monkeypatch.setattr(replace_guard, "MOUNTS", str(found.table))
    _mounted(found)
    return found


def _mounted(dirs: SimpleNamespace, *points: Path) -> None:
    """Write the mount table: the root, and each of points, as Linux writes one (a space is \\040)."""
    lines = ["25 1 259:3 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"]
    for number, point in enumerate(points, 26):
        path = str(point).replace("\\", "\\134").replace(" ", "\\040")
        # The same device as the root: a bind mount, which neither st_dev nor os.path.ismount tells
        lines.append(f"{number} 25 259:3 /library {path} rw,relatime - ext4 /dev/sda1 rw")
    dirs.table.write_text("\n".join(lines) + "\n")


def _place(dirs: SimpleNamespace) -> str:
    return anyio.run(PlacedFolders().place, str(dirs.album), "RED")


@pytest.mark.parametrize("where", [".", "More music", "More music/Deeper"])
def test_a_folder_that_is_or_holds_a_mount_point_is_declined_without_asking(monkeypatch, dirs, where: str) -> None:
    before = contents(dirs.dest)
    _mounted(dirs, dirs.dest / where)
    asked = answer(monkeypatch)

    with pytest.raises(PlacementDeclined, match="mount point"):
        _place(dirs)

    assert asked == []
    assert contents(dirs.dest) == before


def test_a_mount_made_during_the_question_is_not_replaced(monkeypatch, dirs) -> None:
    before = contents(dirs.dest)

    async def mount_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        _mounted(dirs, dirs.inner)
        return True

    monkeypatch.setattr(interaction, "confirm", mount_then_agree)

    with pytest.raises(PlacementDeclined, match="mount point"):
        _place(dirs)

    assert contents(dirs.dest) == before


def test_a_mount_that_went_aside_with_the_folder_is_neither_changed_nor_emptied(monkeypatch, dirs, capsys) -> None:
    dirs.inner.chmod(0o555)
    rename = os.rename

    def moved_with_its_mount(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        rename(src, dst, *args, **kwargs)
        if placement.REPLACED_PREFIX in str(dst):
            _mounted(dirs, Path(dst) / "More music")  # Linux moves a mount with the folder holding it

    monkeypatch.setattr(os, "rename", moved_with_its_mount)
    answer(monkeypatch, True)

    _place(dirs)

    [kept] = dirs.red.glob(".salmon-replaced-*")
    inner = kept / "More music"
    try:
        assert sorted(file.name for file in inner.iterdir()) == ["01.flac", "02.flac"]
        assert inner.stat().st_mode & 0o777 == 0o555
        assert str(kept) in capsys.readouterr().out
    finally:
        inner.chmod(0o755)


def test_a_mount_is_found_in_a_folder_whose_name_the_table_escapes(dirs) -> None:
    folder = album(dirs.red / "My Album")  # A space, which the table writes in octal
    _mounted(dirs, folder / "More")

    assert replace_guard.mount_inside(str(folder)) == str(folder / "More")


def test_mounts_elsewhere_are_no_reason_to_refuse(monkeypatch, dirs) -> None:
    beside = album(dirs.red / "Album 2")
    (dirs.dest / "link").symlink_to(beside, target_is_directory=True)
    _mounted(dirs, beside, dirs.red)  # A name that starts the same, one a link leads to, and one around it
    answer(monkeypatch, True)

    _place(dirs)

    assert contents(dirs.dest) == contents(dirs.album)


@pytest.mark.parametrize("linked", [False, True], ids=["mounted", "behind a link"])
def test_without_a_mount_table_each_folder_is_asked(monkeypatch, dirs, linked: bool) -> None:
    monkeypatch.setattr(replace_guard, "MOUNTS", str(dirs.table.with_name("none")))  # As on macOS and Windows
    mounted = dirs.inner
    if linked:
        mounted = album(dirs.root / "elsewhere")
        shutil.rmtree(dirs.inner)
        dirs.inner.symlink_to(mounted, target_is_directory=True)
    monkeypatch.setattr(os.path, "ismount", lambda path: os.path.realpath(path) == os.path.realpath(mounted))
    answer(monkeypatch, True)

    if linked:
        _place(dirs)
        assert contents(dirs.dest) == contents(dirs.album)
    else:
        with pytest.raises(PlacementDeclined, match="mount point"):
            _place(dirs)
