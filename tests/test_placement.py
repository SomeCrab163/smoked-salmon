"""Placing an album in a tracker's directory (#1056): linked or copied whole, never over what must be kept.

These guard deletion of users' data, so each guard has a case of its own.
"""

import errno
import os
import shutil
import stat
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import asyncclick as click
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import answer, building_left, contents, inodes, links_fail, tracker_dirs

from salmon import cfg, dryrun, hardlinks, interaction, placement
from salmon.converter.conversions import conversion_of, record_conversion
from salmon.placement import PlacedFolders, PlacementDeclined, PlacementError


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album", cover=True)
    (found.album / "Scans").mkdir()
    (found.album / "Scans" / "back.jpg").write_bytes(b"back")
    found.dest = found.red / "Album"
    return found


def _place(dirs: SimpleNamespace, placed: PlacedFolders | None = None, src: Path | None = None) -> str:
    return anyio.run((placed or PlacedFolders()).place, str(src or dirs.album), "RED")


def _earlier(dirs: SimpleNamespace) -> Path:
    """Another folder already at the destination, with times well before now."""
    dirs.dest.mkdir(parents=True)
    (dirs.dest / "other.flac").write_bytes(b"another album")
    os.utime(dirs.dest, ns=(1_000_000_000, 1_000_000_000))
    return dirs.dest


# --- Linking and copying (T1-T5, T9) ---------------------------------------------------------------------------------


@pytest.mark.parametrize("linked", [True, False], ids=["link", "copy"])
def test_the_album_is_linked_or_copied_whole_with_normal_permissions(monkeypatch, dirs, capsys, linked) -> None:
    monkeypatch.setattr(cfg.directory, "hardlinks", linked)
    dirs.album.chmod(0o755)
    asked = answer(monkeypatch)

    assert _place(dirs) == str(dirs.dest)

    assert contents(dirs.dest) == contents(dirs.album)
    assert (inodes(dirs.dest) == inodes(dirs.album)) is linked
    assert stat.S_IMODE(dirs.dest.stat().st_mode) == 0o755  # Not mkdtemp's 0700
    assert asked == []
    assert building_left(dirs.root) == []
    assert f"Created {dirs.red}." in capsys.readouterr().out


def test_a_link_that_fails_part_way_asks_once_then_copies_all(monkeypatch, dirs) -> None:
    links_fail(monkeypatch, after=1)  # Same st_dev, EXDEV from os.link: two bind mounts of one filesystem
    asked: list[tuple[str, dict[str, Any]]] = []
    left_at_question: list[list[str]] = []

    async def confirm(text: str, **kwargs: Any) -> bool:
        asked.append((text, kwargs))
        left_at_question.append([str(f) for temp in building_left(dirs.root) for f in temp.rglob("*")])
        return True

    monkeypatch.setattr(interaction, "confirm", confirm)

    _place(dirs)

    assert len(asked) == 1
    text, options = asked[0]
    assert str(dirs.album) in text and str(dirs.dest) in text and "MB" in text and os.strerror(errno.EXDEV) in text
    assert options == {"default": True, "labels": ("Copy", "Don't copy")}
    assert left_at_question == [[]]  # The link it made is gone before it asks
    assert contents(dirs.dest) == contents(dirs.album)
    assert not set(inodes(dirs.dest).values()) & set(inodes(dirs.album).values())
    assert building_left(dirs.root) == []


@pytest.mark.parametrize(
    ("code", "winerror"),
    [
        (errno.EXDEV, None),
        (errno.EPERM, None),
        (errno.ENOTSUP, None),
        (errno.EOPNOTSUPP, None),
        (errno.EMLINK, None),
        (errno.ENOSYS, None),
        (errno.EINVAL, 1),
    ],
    ids=["EXDEV", "EPERM", "ENOTSUP", "EOPNOTSUPP", "EMLINK", "ENOSYS", "Windows error 1"],
)
def test_each_way_a_link_is_impossible_reaches_the_question(monkeypatch, dirs, code, winerror) -> None:
    links_fail(monkeypatch, code, winerror=winerror)
    asked = answer(monkeypatch, True)

    _place(dirs)

    assert len(asked) == 1
    assert contents(dirs.dest) == contents(dirs.album)


def test_another_device_reaches_the_question_without_trying_a_link(monkeypatch, dirs) -> None:
    real_stat = os.stat
    tried: list[str] = []

    def other_device(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        found = real_stat(path, *args, **kwargs)
        if isinstance(path, str) and os.path.basename(path).startswith(placement.BUILDING_PREFIX):
            return os.stat_result((found[0], found[1], found[2] + 1, *found[3:10]))
        return found

    monkeypatch.setattr(os, "stat", other_device)
    monkeypatch.setattr(os, "link", lambda src, _dst: tried.append(src))
    asked = answer(monkeypatch, True)

    _place(dirs)

    assert tried == []
    assert "different filesystems" in asked[0][0]


def test_only_placement_takes_exdev_for_a_link_it_cannot_make() -> None:
    error = OSError(errno.EXDEV, "Invalid cross-device link")

    assert hardlinks.cannot_link(error)
    assert not hardlinks.no_hard_links(error)  # renames.py's test: unchanged


def test_a_declined_copy_leaves_nothing_and_keeps_the_folder_to_replace(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs))
    links_fail(monkeypatch)
    answer(monkeypatch, True, False)  # Replace it; don't copy

    with pytest.raises(PlacementDeclined, match="Not copying"):
        _place(dirs)

    assert contents(dirs.dest) == before
    assert building_left(dirs.root) == []


def test_yyy_copies_without_asking_and_says_so(monkeypatch, dirs, capsys) -> None:
    links_fail(monkeypatch)
    asked = answer(monkeypatch)

    with interaction.assuming_defaults():
        _place(dirs)

    assert asked == []
    assert "-yyy: copying" in capsys.readouterr().out
    assert contents(dirs.dest) == contents(dirs.album)


def test_stopping_at_the_copy_question_leaves_nothing_half_built(monkeypatch, dirs) -> None:
    links_fail(monkeypatch)

    async def stop(*_args: Any, **_kwargs: Any) -> bool:
        raise click.Abort

    monkeypatch.setattr(interaction, "confirm", stop)

    with pytest.raises(click.Abort):
        _place(dirs)

    assert not dirs.dest.exists()
    assert building_left(dirs.root) == []


@pytest.mark.parametrize("code", [errno.EACCES, errno.ENOSPC])
@pytest.mark.parametrize("step", ["link", "copy"])
def test_any_other_error_fails_and_keeps_the_folder_to_replace(monkeypatch, dirs, code, step) -> None:
    before = contents(_earlier(dirs))

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(code, os.strerror(code))

    if step == "link":
        links_fail(monkeypatch, code, after=1)
    else:
        monkeypatch.setattr(cfg.directory, "hardlinks", False)
        monkeypatch.setattr(shutil, "copy2", fail)
    asked = answer(monkeypatch, True)  # Only the question to replace

    with pytest.raises(PlacementError, match=os.strerror(code)) as raised:
        _place(dirs)

    assert not isinstance(raised.value, PlacementDeclined)
    assert str(dirs.album) in str(raised.value) and str(dirs.dest) in str(raised.value)
    assert len(asked) == 1
    assert contents(dirs.dest) == before
    assert building_left(dirs.root) == []


def test_a_directory_that_cannot_be_created_fails_the_placement(dirs) -> None:
    dirs.red.parent.write_bytes(b"a file where the parent should be")

    with pytest.raises(PlacementError, match="Could not create"):
        _place(dirs)


def test_symlinks_become_regular_files_and_folders(monkeypatch, dirs) -> None:
    outside = album(dirs.root / "outside", tracks=1)
    (dirs.album / "linked.flac").symlink_to(outside / "01.flac")
    (dirs.album / "relative.jpg").symlink_to("cover.jpg")
    (dirs.album / "linked folder").symlink_to(outside, target_is_directory=True)
    answer(monkeypatch)

    _place(dirs)

    assert [entry for entry in dirs.dest.rglob("*") if entry.is_symlink()] == []
    assert (dirs.dest / "linked.flac").stat().st_ino == (outside / "01.flac").stat().st_ino
    assert (dirs.dest / "linked folder" / "01.flac").is_file()


@pytest.mark.parametrize("how", ["album in it", "album holds it", "file linked into it", "folder linked into it"])
def test_what_reaches_a_library_is_copied_without_asking(monkeypatch, dirs, how) -> None:
    source, library = dirs.album, dirs.library
    if how == "album in it":
        source = album(library / "Album")
    elif how == "album holds it":
        library = dirs.album / "Scans"
    elif how == "file linked into it":
        (source / "linked.flac").symlink_to(album(library / "Other") / "01.flac")
    else:
        (source / "linked").symlink_to(album(library / "Other"), target_is_directory=True)
    monkeypatch.setattr(cfg.directory, "library_dirs", [str(library)])
    asked = answer(monkeypatch)

    _place(dirs, src=source)

    assert asked == []
    assert not set(inodes(dirs.dest).values()) & {file.stat().st_ino for file in source.rglob("*") if file.is_file()}


def test_a_read_only_album_is_placed_read_only(dirs) -> None:
    (dirs.album / "Scans").chmod(0o555)
    dirs.album.chmod(0o555)
    try:
        _place(dirs)

        assert stat.S_IMODE(dirs.dest.stat().st_mode) == 0o555
        assert stat.S_IMODE((dirs.dest / "Scans").stat().st_mode) == 0o555
        assert building_left(dirs.root) == []
    finally:
        for folder in (dirs.album, dirs.dest):
            if folder.exists():
                folder.chmod(0o755)
                (folder / "Scans").chmod(0o755)


def test_a_half_built_read_only_folder_is_removed(monkeypatch, dirs) -> None:
    (dirs.album / "Scans").chmod(0o555)
    links_fail(monkeypatch, errno.EACCES, after=2)
    try:
        with pytest.raises(PlacementError):
            _place(dirs)

        assert building_left(dirs.root) == []
    finally:
        (dirs.album / "Scans").chmod(0o755)


# --- An existing destination (T6, T7) --------------------------------------------------------------------------------


def test_declining_to_replace_changes_nothing(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs))
    asked = answer(monkeypatch, False)

    with pytest.raises(PlacementDeclined):
        _place(dirs)

    assert asked[0][1]["default"] is False
    assert contents(dirs.dest) == before


def test_agreeing_replaces_and_never_merges(monkeypatch, dirs) -> None:
    _earlier(dirs)
    answer(monkeypatch, True)

    _place(dirs)

    assert contents(dirs.dest) == contents(dirs.album)  # No other.flac


def test_yyy_never_replaces_and_never_asks(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs))
    asked = answer(monkeypatch)

    with interaction.assuming_defaults(), pytest.raises(PlacementDeclined, match="-yyy"):
        _place(dirs)

    assert asked == []
    assert contents(dirs.dest) == before


def test_the_question_says_when_a_torrent_may_seed_from_the_folder(monkeypatch, dirs) -> None:
    os.link(_earlier(dirs) / "other.flac", dirs.root / "seeded.flac")
    asked = answer(monkeypatch, False)

    with pytest.raises(PlacementDeclined):
        _place(dirs)

    assert "a torrent may seed from it" in asked[0][0]


def _recreated(dirs: SimpleNamespace) -> None:
    shutil.rmtree(dirs.dest)
    dirs.dest.mkdir()  # On ext4 this has the deleted folder's inode
    (dirs.dest / "new.flac").write_bytes(b"new")


def _another_folder_there(dirs: SimpleNamespace) -> None:
    dirs.dest.rename(dirs.red / "moved away")
    dirs.dest.mkdir()
    os.utime(dirs.dest, ns=(1_000_000_000, 1_000_000_000))


def _entry_added(dirs: SimpleNamespace) -> None:
    (dirs.dest / "added.flac").write_bytes(b"added")  # Same inode, as a recreated folder's can be


def _now_shared(dirs: SimpleNamespace) -> None:
    os.link(dirs.dest / "other.flac", dirs.root / "seeded.flac")  # Changes neither the folder's inode nor its times


def _linked_elsewhere(dirs: SimpleNamespace) -> None:
    dirs.dest.rename(dirs.red / "moved away")
    dirs.dest.symlink_to(dirs.red / "moved away", target_is_directory=True)


@pytest.mark.parametrize("change", [_recreated, _another_folder_there, _entry_added, _now_shared, _linked_elsewhere])
def test_a_folder_that_changes_during_the_question_is_not_replaced(monkeypatch, dirs, change) -> None:
    _earlier(dirs)
    changed: dict[str, str] = {}

    async def change_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        change(dirs)
        changed.update(contents(dirs.dest))
        return True

    monkeypatch.setattr(interaction, "confirm", change_then_agree)

    with pytest.raises(PlacementDeclined):
        _place(dirs)

    assert contents(dirs.dest) == changed
    assert building_left(dirs.root) == []


def test_a_folder_that_goes_away_during_the_question_is_not_made_again(monkeypatch, dirs) -> None:
    _earlier(dirs)

    async def remove_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        shutil.rmtree(dirs.dest)
        return True

    monkeypatch.setattr(interaction, "confirm", remove_then_agree)

    with pytest.raises(PlacementDeclined, match="went away"):
        _place(dirs)


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "with files"])
def test_a_folder_that_appears_during_the_question_is_not_replaced(monkeypatch, dirs, empty) -> None:
    links_fail(monkeypatch)
    appeared: list[int] = []

    async def appear_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        dirs.dest.mkdir()
        if not empty:
            (dirs.dest / "other.flac").write_bytes(b"other")
        appeared.append(dirs.dest.stat().st_ino)
        return True

    monkeypatch.setattr(interaction, "confirm", appear_then_agree)

    with pytest.raises(PlacementDeclined, match="appeared"):
        _place(dirs)

    assert [dirs.dest.stat().st_ino] == appeared
    assert building_left(dirs.root) == []


@pytest.mark.parametrize("question", ["replace", "copy"])
def test_a_folder_that_becomes_a_data_directory_during_a_question_is_not_replaced(monkeypatch, dirs, question) -> None:
    before = contents(_earlier(dirs))
    links_fail(monkeypatch)

    async def configure_then_agree(text: str, **_kwargs: Any) -> bool:
        if ("Copy instead" in text) == (question == "copy"):
            monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.dest))
        return True

    monkeypatch.setattr(interaction, "confirm", configure_then_agree)

    with pytest.raises(PlacementDeclined, match="which is a download_directory"):
        _place(dirs)

    assert contents(dirs.dest) == before


# --- Never replaced (T8) ---------------------------------------------------------------------------------------------


def _data_directory(monkeypatch, dirs: SimpleNamespace) -> list[str]:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.dest))
    return []


def _holds_data_directory(monkeypatch, dirs: SimpleNamespace) -> list[str]:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.dest / "ops"))  # Not made yet
    return []


def _holds_the_global_directory(monkeypatch, dirs: SimpleNamespace) -> list[str]:
    monkeypatch.setattr(cfg.directory, "download_directory", str(dirs.dest / "dl"))
    return []


def _is_kept(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    return [str(dirs.dest)]


def _holds_kept(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    return [str(dirs.dest / "CD1")]


def _in_kept(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    return [str(dirs.red)]


def _symlink(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    dirs.dest.rename(dirs.red / "real")
    dirs.dest.symlink_to(dirs.red / "real", target_is_directory=True)
    return []


def _dangling(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    shutil.rmtree(dirs.dest)
    dirs.dest.symlink_to(dirs.red / "gone")
    return []


def _not_a_directory(_monkeypatch, dirs: SimpleNamespace) -> list[str]:
    shutil.rmtree(dirs.dest)
    dirs.dest.write_bytes(b"a file")
    return []


def _library(monkeypatch, dirs: SimpleNamespace) -> list[str]:
    monkeypatch.setattr(cfg.directory, "library_dirs", [str(dirs.dest)])
    return []


def _holds_library(monkeypatch, dirs: SimpleNamespace) -> list[str]:
    (dirs.dest / "lib").mkdir()
    monkeypatch.setattr(cfg.directory, "library_dirs", [str(dirs.dest / "lib")])
    return []


Arrange = Callable[[Any, SimpleNamespace], list[str]]


@pytest.mark.parametrize(
    "arrange",
    [
        _data_directory,
        _holds_data_directory,
        _holds_the_global_directory,
        _is_kept,
        _holds_kept,
        _in_kept,
        _symlink,
        _dangling,
        _not_a_directory,
        _library,
        _holds_library,
    ],
)
def test_what_must_never_be_replaced_is_declined_without_asking(monkeypatch, dirs, arrange: Arrange) -> None:
    _earlier(dirs)
    keep = arrange(monkeypatch, dirs)
    before = contents(dirs.red)
    asked = answer(monkeypatch)

    with pytest.raises(PlacementDeclined):
        _place(dirs, PlacedFolders(*keep))

    assert asked == []
    assert contents(dirs.red) == before


def test_a_destination_not_there_yet_whose_parent_leads_into_a_library_is_declined(monkeypatch, dirs) -> None:
    dirs.red.parent.mkdir()
    dirs.red.symlink_to(dirs.library, target_is_directory=True)
    monkeypatch.setattr(cfg.directory, "library_dirs", [str(dirs.library)])

    with pytest.raises(PlacementDeclined, match="library_dirs"):
        _place(dirs)

    assert list(dirs.library.iterdir()) == []


@pytest.mark.parametrize("way", ["inside the album", "holding the album"])
def test_a_destination_that_overlaps_the_album_is_declined(monkeypatch, dirs, way) -> None:
    if way == "inside the album":
        monkeypatch.setattr(cfg.tracker.red, "download_directory", str(dirs.album / "red"))
        source = dirs.album
    else:
        source = album(dirs.red / "Album" / "Album")
    asked = answer(monkeypatch)

    with pytest.raises(PlacementDeclined):
        _place(dirs, src=source)

    assert asked == []
    assert len(list(source.glob("*.flac"))) == 2


@pytest.mark.parametrize("inside", [True, False], ids=["inside", "holding"])
def test_place_folder_itself_refuses_an_overlap(dirs, inside) -> None:
    src, dest = (dirs.album, dirs.album / "New") if inside else (dirs.album, dirs.downloads)
    before = contents(dirs.downloads)

    with pytest.raises(PlacementDeclined):
        anyio.run(partial(placement.place_folder, str(src), str(dest)))

    assert contents(dirs.downloads) == before


def test_a_folder_salmon_web_cannot_claim_is_declined_before_asking(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs))
    asked = answer(monkeypatch)
    claimed: list[Path] = []

    async def claim(folder: str) -> str:
        claimed.append(Path(folder))
        return "another job works on it"

    with placement.claiming(claim), pytest.raises(PlacementDeclined, match="another job"):
        _place(dirs)

    assert claimed == [dirs.dest]
    assert asked == []
    assert contents(dirs.dest) == before


# --- What a run placed -----------------------------------------------------------------------------------------------


def test_the_album_already_there_is_used_as_it_is(monkeypatch, dirs) -> None:
    source = album(dirs.dest)
    asked = answer(monkeypatch)

    assert _place(dirs, src=source) == str(source)
    assert asked == []


def test_a_run_reuses_what_it_placed_and_never_replaces_it_from_another_folder(monkeypatch, dirs) -> None:
    placed = PlacedFolders()
    other = album(dirs.root / "other" / "Album", tags={"album": "Other"})
    asked = answer(monkeypatch)

    assert _place(dirs, placed) == _place(dirs, placed) == str(dirs.dest)
    with pytest.raises(PlacementDeclined, match="this run made it"):
        _place(dirs, placed, other)

    assert asked == []
    assert contents(dirs.dest) == contents(dirs.album)


def test_two_spellings_of_one_directory_give_one_folder_path(monkeypatch, dirs) -> None:
    dirs.red.mkdir(parents=True)
    (dirs.root / "also red").symlink_to(dirs.red, target_is_directory=True)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.root / "also red"))
    placed = PlacedFolders()

    first = _place(dirs, placed)

    assert anyio.run(placed.place, str(dirs.album), "OPS") == first


def test_a_conversion_the_run_made_is_never_replaced(monkeypatch, dirs) -> None:
    made = album(dirs.dest, tags={"album": "the conversion"})
    before = contents(made)
    placed = PlacedFolders()
    placed.made_conversion("an album", "V0", str(made))
    asked = answer(monkeypatch)

    with pytest.raises(PlacementDeclined, match="this run made it"):
        _place(dirs, placed)

    assert asked == []
    assert contents(made) == before


def test_the_conversion_record_follows_and_a_replaced_folders_goes(monkeypatch, dirs) -> None:
    record_conversion(str(_earlier(dirs)), source="/of/the/old", kind="transcode", bitrate="320")
    answer(monkeypatch, True)

    _place(dirs)

    assert conversion_of(str(dirs.dest)) is None

    converted = album(dirs.downloads / "Album [V0]")
    record_conversion(str(converted), source=str(dirs.album), kind="transcode", bitrate="V0")
    _place(dirs, src=converted)

    record = conversion_of(str(dirs.red / "Album [V0]"))
    assert record is not None and (record["source"], record["bitrate"]) == (str(dirs.album), "V0")


def test_without_a_key_nothing_is_placed(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)

    assert _place(dirs, src=album(dirs.root / "elsewhere" / "Album")) == str(dirs.root / "elsewhere" / "Album")


@pytest.mark.parametrize("exists", [True, False], ids=["there", "not there"])
def test_a_dry_run_makes_nothing_and_asks_nothing(monkeypatch, dirs, capsys, exists) -> None:
    if exists:
        _earlier(dirs)
    before = contents(dirs.root)
    asked = answer(monkeypatch)

    with dryrun.mode():
        assert _place(dirs) == str(dirs.album)

    assert asked == []
    assert contents(dirs.root) == before
    assert ("a real run would ask" in capsys.readouterr().out) is exists


# --- A replace that fails part way (D1) ------------------------------------------------------------------------------


def _earlier_with_folders(dirs: SimpleNamespace) -> Path:
    dest = album(dirs.dest, tags={"album": "the old one"})
    for name in ("Artwork", "Booklet"):
        (dest / name).mkdir()
        for number in range(3):
            (dest / name / f"{number}.jpg").write_bytes(name.encode())
    return dest


def _replaced_left(root: Path) -> list[Path]:
    return list(root.rglob(".salmon-replaced-*"))


@pytest.mark.parametrize("read_only", ["a subfolder", "the folder itself"])
def test_a_read_only_folder_can_be_replaced(monkeypatch, dirs, read_only) -> None:
    dest = _earlier_with_folders(dirs)
    (dest / "Booklet" if read_only == "a subfolder" else dest).chmod(0o555)
    answer(monkeypatch, True)

    _place(dirs)

    assert contents(dirs.dest) == contents(dirs.album)
    assert _replaced_left(dirs.root) == []


def test_a_replaced_folder_that_cannot_all_be_removed_is_left_and_named(monkeypatch, dirs, capsys) -> None:
    _earlier_with_folders(dirs)
    unlink = os.unlink

    def not_the_owner(path: Any, *args: Any, **kwargs: Any) -> None:
        if os.fspath(path).endswith("2.jpg"):
            raise PermissionError(errno.EPERM, "Operation not permitted")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", not_the_owner)
    answer(monkeypatch, True)

    assert _place(dirs) == str(dirs.dest)

    assert contents(dirs.dest) == contents(dirs.album)
    [left] = _replaced_left(dirs.root)
    assert str(left) in capsys.readouterr().out
    assert building_left(dirs.root) == []


@pytest.mark.parametrize("stopped", [OSError(errno.EIO, "I/O error"), KeyboardInterrupt()], ids=["error", "Ctrl-C"])
@pytest.mark.parametrize("when", ["before the move", "after the move"])
def test_a_replace_whose_last_move_fails_puts_the_old_folder_back(monkeypatch, dirs, stopped, when) -> None:
    before = contents(_earlier_with_folders(dirs))
    rename = os.rename
    failed: list[str] = []

    def fail_into_place(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if placement.BUILDING_PREFIX in src and not failed:
            failed.append(src)
            if when == "after the move":  # As a signal that arrives as the rename returns
                rename(src, dst, *args, **kwargs)
            raise stopped
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", fail_into_place)
    answer(monkeypatch, True)

    with pytest.raises((PlacementError, KeyboardInterrupt)):
        _place(dirs)

    assert contents(dirs.dest) == before
    assert _replaced_left(dirs.root) == []
    assert building_left(dirs.root) == []


def test_a_folder_that_cannot_be_moved_aside_is_untouched(monkeypatch, dirs) -> None:
    before = contents(_earlier_with_folders(dirs))
    rename = os.rename

    def in_use(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if src == str(dirs.dest):
            raise OSError(errno.EBUSY, "Device or resource busy")  # A mount point, or open on Windows
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", in_use)
    answer(monkeypatch, True)

    with pytest.raises(PlacementError):
        _place(dirs)

    assert contents(dirs.dest) == before


def test_an_old_folder_that_cannot_be_put_back_is_kept_and_named(monkeypatch, dirs, capsys) -> None:
    before = contents(_earlier_with_folders(dirs))
    rename = os.rename

    def only_the_first(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if src != str(dirs.dest):
            raise OSError(errno.EIO, "I/O error")
        rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", only_the_first)
    answer(monkeypatch, True)

    with pytest.raises(PlacementError):
        _place(dirs)

    [kept] = _replaced_left(dirs.root)
    assert contents(kept) == before
    assert str(kept) in capsys.readouterr().out


def test_a_folder_that_appears_as_the_old_one_is_moved_aside_is_not_replaced(monkeypatch, dirs, capsys) -> None:
    before = contents(_earlier_with_folders(dirs))
    rename = os.rename

    def then_one_appears(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        rename(src, dst, *args, **kwargs)
        if src == str(dirs.dest):
            dirs.dest.mkdir()  # Empty: os.rename would silently replace it

    monkeypatch.setattr(os, "rename", then_one_appears)
    answer(monkeypatch, True)

    with pytest.raises(PlacementDeclined, match="appeared"):
        _place(dirs)

    assert contents(dirs.dest) == {}
    [kept] = _replaced_left(dirs.root)
    assert contents(kept) == before
    assert str(kept) in capsys.readouterr().out


def test_removing_the_old_folder_changes_nothing_a_link_in_it_leads_to(monkeypatch, dirs) -> None:
    outside = album(dirs.library / "Album")
    inner = outside / "Scans"  # What a walk through the link would reach and make writable
    inner.mkdir()
    inner.chmod(0o555)
    outside.chmod(0o555)
    try:
        dest = _earlier_with_folders(dirs)
        (dest / "linked").symlink_to(outside, target_is_directory=True)
        answer(monkeypatch, True)

        _place(dirs)

        assert [stat.S_IMODE(folder.stat().st_mode) for folder in (outside, inner)] == [0o555, 0o555]
        assert sorted(file.name for file in outside.iterdir()) == ["01.flac", "02.flac", "Scans"]
    finally:
        outside.chmod(0o755)
        inner.chmod(0o755)


def test_a_folder_that_cannot_be_removed_keeps_its_group_and_others_permissions(monkeypatch, dirs) -> None:
    dest = _earlier_with_folders(dirs)
    (dest / "Booklet").chmod(0o555)

    def not_the_owner(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "unlink", not_the_owner)
    answer(monkeypatch, True)

    _place(dirs)

    [left] = _replaced_left(dirs.root)
    assert stat.S_IMODE((left / "Booklet").stat().st_mode) == 0o755


def _failing_at(real: Callable[..., Any], failing: int, calls: list[int]) -> Callable[..., Any]:
    """`real`, with its call number `failing` raising EIO."""

    def call(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == failing:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real(*args, **kwargs)

    return call


@pytest.mark.parametrize("linked", [True, False], ids=["link", "copy"])
@pytest.mark.parametrize(
    "call", ["rename", "chmod", "mkdir", "link", "unlink", "rmdir", "lstat", "scandir"], ids=lambda call: f"os.{call}"
)
def test_whichever_call_fails_the_destination_is_the_old_folder_or_the_new(
    monkeypatch, tmp_path, linked: bool, call: str
) -> None:
    real = getattr(os, call)
    failing = 0
    while True:  # Each of this call that a replace makes fails in a run of its own
        failing += 1
        root = tmp_path / str(failing)
        root.mkdir()
        found = tracker_dirs(monkeypatch, root)
        monkeypatch.setattr(cfg.directory, "hardlinks", linked)
        found.album, found.dest = album(found.downloads / "Album", cover=True), found.red / "Album"
        old, new = contents(_earlier_with_folders(found)), contents(found.album)
        answer(monkeypatch, True, True)
        calls: list[int] = []

        with monkeypatch.context() as faulty:
            faulty.setattr(os, call, _failing_at(real, failing, calls))
            try:
                _place(found)
            except (PlacementError, OSError):
                outcome = [old]
            else:
                outcome = [new]

        assert contents(found.dest) in outcome, f"call {failing}"
        if len(calls) < failing:
            return


# --- The conversion record is part of a placement --------------------------------------------------------------------


def _read_only_records(parent: Path) -> Path:
    records = parent / ".salmon-conversions"
    records.mkdir(parents=True, exist_ok=True)
    records.chmod(0o555)  # As a container's other user leaves it
    return records


@pytest.mark.parametrize("replaces", [True, False], ids=["replacing", "new"])
def test_a_record_that_cannot_be_written_undoes_the_placement(monkeypatch, dirs, replaces) -> None:
    source = album(dirs.downloads / "Album [V0]")
    dest = dirs.red / source.name
    record_conversion(str(source), source="/the/album", kind="transcode", bitrate="V0")
    before = None
    if replaces:
        album(dest, tags={"album": "the old one"})
        record_conversion(str(dest), source="/the/old", kind="transcode", bitrate="320")
        before = contents(dest), conversion_of(str(dest))
        answer(monkeypatch, True)
        # An existing record is replaced by a file beside it, which a read-only folder refuses.
    records = _read_only_records(dirs.red)
    try:
        with pytest.raises(PlacementError, match="Could not make") as failed:
            _place(dirs, src=source)
    finally:
        records.chmod(0o755)

    assert not isinstance(failed.value, PlacementDeclined)
    if before is None:
        assert not dest.exists()  # So a retry doesn't find it
        assert _place(dirs, src=source) == str(dest)
    else:
        assert (contents(dest), conversion_of(str(dest))) == before
    assert building_left(dirs.root) == []
    assert _replaced_left(dirs.root) == []


def test_a_record_that_cannot_be_read_undoes_the_placement(dirs) -> None:
    source = album(dirs.downloads / "Album [V0]")
    record_conversion(str(source), source="/the/album", kind="transcode", bitrate="V0")
    sidecar = dirs.downloads / ".salmon-conversions" / "Album [V0].json"
    sidecar.chmod(0o000)
    try:
        with pytest.raises(PlacementError):
            _place(dirs, src=source)
    finally:
        sidecar.chmod(0o644)

    assert not (dirs.red / source.name).exists()


def test_a_stale_record_at_the_name_is_cleared(dirs) -> None:
    dirs.red.mkdir(parents=True)
    # Of a folder deleted by hand since
    record_conversion(str(dirs.dest), source="/gone", kind="transcode", bitrate="V0")

    _place(dirs)

    assert conversion_of(str(dirs.dest)) is None  # A FLAC, never described as a transcode


def test_a_stale_record_that_cannot_be_cleared_undoes_the_placement(dirs) -> None:
    dirs.red.mkdir(parents=True)
    record_conversion(str(dirs.dest), source="/gone", kind="transcode", bitrate="V0")
    records = _read_only_records(dirs.red)
    try:
        with pytest.raises(PlacementError):
            _place(dirs)
    finally:
        records.chmod(0o755)

    assert not dirs.dest.exists()


def test_an_album_without_a_record_needs_no_writable_record_folder(dirs) -> None:
    records = _read_only_records(dirs.red)
    try:
        assert _place(dirs) == str(dirs.dest)
    finally:
        records.chmod(0o755)
