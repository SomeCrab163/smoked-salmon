"""Seedboxes with per-tracker directories (#1056): each client's save path, and what may be removed."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import anyio
import pytest
from helpers.album_files import album
from helpers.tracker_dirs import tracker_dirs

from salmon import cfg
from salmon.config.validations import Seedbox
from salmon.uploader import seedbox as sb


@pytest.fixture
def world(monkeypatch, tmp_path) -> SimpleNamespace:
    """Three trackers, RED and OPS with a folder of their own, and fake rclone and torrent clients."""
    found = tracker_dirs(monkeypatch, tmp_path)
    found.saved = []  # (seedbox, save path, torrent) given to a client
    monkeypatch.setattr(sb.TorrentClientGenerator, "parse_libtc_url", staticmethod(lambda _url: object()))

    async def copy(*_args) -> bool:
        return True

    async def add(_client, shell_path, torrent, _label, _paused, _secrets, name) -> bool:
        found.saved.append((name, shell_path, Path(torrent).stem))
        return True

    monkeypatch.setattr(sb, "_rclone_upload_folder", copy)
    monkeypatch.setattr(sb, "_add_to_downloader", add)
    return found


def _box(name: str, kind: Literal["local", "rclone"] = "local", **settings) -> Seedbox:
    return Seedbox(name=name, type=kind, torrent_client=f"fake://{name}", **settings)


def _seed(manager: sb.UploadManager, folder: Path, code: str) -> None:
    manager.add_upload_task(str(folder), task_type="folder", is_flac=True, site_code=code)
    manager.add_upload_task(f"/torrents/{code}.torrent", "seed", True, folder=str(folder), site_code=code)


def _run(monkeypatch, boxes: list[Seedbox], *queued: tuple[Path, str]) -> sb.UploadManager:
    monkeypatch.setattr(cfg, "seedbox", boxes)
    manager = sb.UploadManager()
    for folder, code in queued:
        _seed(manager, folder, code)
    anyio.run(manager.execute_upload)
    return manager


def test_a_local_client_gets_each_trackers_folder(monkeypatch, world) -> None:
    queued = ((world.red / "A", "RED"), (world.ops / "A", "OPS"), (world.downloads / "A", "DIC"))

    _run(monkeypatch, [_box("home")], *queued)

    assert sorted(world.saved) == sorted(
        [("home", str(world.red), "RED"), ("home", str(world.ops), "OPS"), ("home", str(world.downloads), "DIC")]
    )


def test_a_relative_folder_is_given_absolute(monkeypatch, world) -> None:
    monkeypatch.chdir(world.root)
    monkeypatch.setattr(cfg.tracker.red, "download_directory", "data/red")

    _run(monkeypatch, [_box("home")], (world.red / "A", "RED"))

    assert world.saved == [("home", str(Path(os.getcwd(), "data", "red")), "RED")]


@pytest.mark.parametrize("kind", ["local", "rclone"])
def test_a_seedboxs_own_directory_wins(monkeypatch, world, kind: Literal["local", "rclone"]) -> None:
    _run(monkeypatch, [_box("box", kind, url="nas", directory="/client/path")], (world.red / "A", "RED"))

    assert world.saved == [("box", "/client/path", "RED")]


def test_rclone_without_a_directory_is_unchanged(monkeypatch, world) -> None:
    _run(monkeypatch, [_box("box", "rclone", url="nas")], (world.red / "A", "RED"))

    assert world.saved == [("box", "", "RED")]


def test_the_local_copy_in_a_trackers_folder_is_removed_once_seeding_elsewhere(monkeypatch, world) -> None:
    folder = album(world.red / "Album")
    box = _box("nas", "rclone", url="nas", remove_local_after_copy=True)

    _run(monkeypatch, [box], (folder, "RED")).remove_local_copies()

    assert not folder.exists()
    assert world.red.is_dir()


def test_each_trackers_folder_is_removed_or_kept_on_its_own(monkeypatch, world) -> None:
    red = album(world.red / "Album")
    world.ops.mkdir(parents=True)
    ops = Path(shutil.copytree(red, world.ops / "Album", copy_function=os.link))
    boxes = [
        _box("nas", "rclone", url="nas", trackers=["RED"], remove_local_after_copy=True),
        _box("home", trackers=["OPS"]),
    ]

    _run(monkeypatch, boxes, (red, "RED"), (ops, "OPS")).remove_local_copies()

    assert not red.exists()
    assert sorted(file.name for file in ops.iterdir()) == ["01.flac", "02.flac"]


@pytest.mark.parametrize("which", ["the trackers folder", "a folder holding it", "the global one", "outside them all"])
def test_a_data_directory_or_what_holds_one_is_never_removed(monkeypatch, world, which) -> None:
    if which == "the trackers folder":
        # Inside the global directory: being inside that is not enough.
        folder = world.downloads / "red"
        monkeypatch.setattr(cfg.tracker.red, "download_directory", str(folder))
    elif which == "a folder holding it":
        folder = world.downloads / "trackers"
        monkeypatch.setattr(cfg.tracker.red, "download_directory", str(folder / "red"))
    elif which == "the global one":
        folder = world.root / "outer" / "downloads"  # Inside OPS's, and the global directory itself
        monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(world.root / "outer"))
        monkeypatch.setattr(cfg.directory, "download_directory", str(folder))
    else:
        folder = world.root / "elsewhere"
    album(folder / "Seeding")
    box = _box("nas", "rclone", url="nas", remove_local_after_copy=True)

    _run(monkeypatch, [box], (folder, "RED")).remove_local_copies()

    assert (folder / "Seeding" / "01.flac").is_file()
