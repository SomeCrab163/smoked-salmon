"""Guards of placement and its claim that no other test saw switched off (#1056). They guard deletion of users' data."""

import concurrent.futures
import errno
import os
import shutil
import stat
from functools import partial
from types import SimpleNamespace
from typing import Any

import anyio
import msgspec
import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album
from helpers.tracker_dirs import answer, contents, tracker_dirs
from helpers.uploader_dry_run import METADATA, _returning
from helpers.webui_jobs import _finished, _question, _with_app

import salmon.tagger.foldername
from salmon import cfg, interaction, placement
from salmon.converter import conversions
from salmon.converter.conversions import conversion_of, record_conversion
from salmon.placement import PlacedFolders, PlacementDeclined, PlacementError
from salmon.tagger.foldername import rename_folder
from salmon.trackers import base
from salmon.webui import jobs
from salmon.webui.jobs import JobKind, JobManager


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album")
    found.dest = found.red / "Album"
    return found


def test_an_undo_moves_out_only_the_folder_it_put_there(monkeypatch, dirs) -> None:
    album(dirs.dest, tags={"album": "the one there"})
    third: dict[str, str] = {}

    def another_takes_the_name(*_args: Any) -> None:
        dirs.dest.rename(dirs.red / "moved away")
        third.update(contents(album(dirs.dest, tags={"album": "a third"})))
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(conversions, "copy_record", another_takes_the_name)
    answer(monkeypatch, True)

    with pytest.raises(PlacementError):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    assert contents(dirs.dest) == third  # Moved to the build folder, it would have been deleted with it


def test_place_folder_claims_before_it_builds(monkeypatch, dirs) -> None:
    built: list[str] = []
    monkeypatch.setattr(shutil, "copytree", lambda src, *_args, **_kwargs: built.append(src))

    async def refuse(_folder: str) -> str:
        return "another job works on it"

    with placement.claiming(refuse), pytest.raises(PlacementDeclined):
        anyio.run(placement.place_folder, str(dirs.album), str(dirs.dest))

    assert built == []
    assert not dirs.red.exists()


def test_another_folder_there_during_the_last_claim_is_not_replaced(monkeypatch, dirs) -> None:
    album(dirs.dest, tags={"album": "the one there"})
    third: dict[str, str] = {}
    claims: list[str] = []

    async def claim(folder: str) -> None:
        claims.append(folder)
        if len(claims) == 3:  # Before the question, before the build, and just before the move
            dirs.dest.rename(dirs.red / "moved away")
            third.update(contents(album(dirs.dest, tags={"album": "a third"})))

    answer(monkeypatch, True)

    with placement.claiming(claim), pytest.raises(PlacementDeclined, match="it changed"):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    assert contents(dirs.dest) == third


def test_a_move_that_fails_gives_the_folder_its_mode_back(monkeypatch, dirs) -> None:
    dirs.album.chmod(0o555)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "rename", fail)
    try:
        with pytest.raises(OSError):
            placement._move(str(dirs.album), str(dirs.root / "to"))
        assert stat.S_IMODE(dirs.album.stat().st_mode) == 0o555
    finally:
        dirs.album.chmod(0o755)


def test_a_junction_is_never_replaced(monkeypatch, dirs) -> None:
    album(dirs.dest)
    monkeypatch.setattr(os.path, "isjunction", lambda path: str(path) == str(dirs.dest))  # Only Windows has them

    assert placement.never_replaced(str(dirs.dest)) == "it is a link"


def test_a_rename_into_the_folder_itself_carries_its_record(monkeypatch, dirs) -> None:
    folder = album(dirs.red / "New")
    given = dirs.root / "in" / "Given"  # Another path to the same folder
    given.parent.mkdir()
    given.symlink_to(folder, target_is_directory=True)
    record_conversion(str(given), source="/the/album", kind="transcode", bitrate="V0")
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", _returning("New"))
    answer(monkeypatch)

    anyio.run(partial(rename_folder, str(given), dict(METADATA), True, parent=str(dirs.red)))

    assert (conversion_of(str(folder)) or {}).get("source") == "/the/album"


def test_declined_conversions_are_given_once() -> None:
    placed = PlacedFolders()
    placed.conversion_failed("MP3 V0", PlacementDeclined("declined"))

    assert placed.take_declined_conversions() == ["MP3 V0"]
    assert placed.take_declined_conversions() == []  # Else the next release is blamed for it


# --- salmon web's claim ----------------------------------------------------------------------------------------------


class Folder(msgspec.Struct):
    folder: str


@pytest.fixture
def web(monkeypatch, dirs, tmp_path) -> SimpleNamespace:
    """A job kind that claims its folder as a destination and holds it until it is answered."""
    torrents = tmp_path / "torrents"
    torrents.mkdir()
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(torrents))
    monkeypatch.setattr(base, "_learned_secrets", set())

    async def claim(params: Folder) -> None:
        await placement.claim(params.folder)
        await interaction.confirm("Claimed. End?")

    async def nothing(_params: Folder) -> None:
        pass

    for name, run, held in (("claim", claim, False), ("held", nothing, True)):
        folder = (lambda params: params.folder) if held else (lambda _params: None)
        kind = JobKind(name=name, params=Folder, run=run, title=lambda _params: "a job", folder=folder)
        monkeypatch.setitem(jobs.KINDS, name, kind)
    return dirs


def test_a_job_started_on_a_claimed_folder_waits_however_the_claim_spelled_it(web) -> None:
    real = album(web.dest)
    spelled = web.root / "red link"
    spelled.symlink_to(web.red, target_is_directory=True)

    async def test(_client: TestClient, manager: JobManager) -> None:
        claiming = manager.start("claim", {"folder": str(spelled / "Album")})
        await _question(manager, claiming.id)
        waiting = manager.start("held", {"folder": str(real)})
        assert waiting.status == "queued"
        manager.cancel(claiming.id)
        await _finished(manager, waiting.id)

    _with_app(test)


def test_an_overlap_is_seen_through_a_second_mount_of_the_folder(monkeypatch, web) -> None:
    real = album(web.dest)
    mounted = web.root / "mounted again"
    stat_of = os.stat

    def stat_through_the_mount(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if isinstance(path, str | os.PathLike):
            path = str(path).replace(str(mounted), str(web.red))  # The same device and inode under another path
        return stat_of(path, *args, **kwargs)

    kind = jobs.KINDS["held"]
    job = jobs.Job(kind, Folder(str(real)), {}, "a job", str(real), False, False)
    monkeypatch.setattr(os, "stat", stat_through_the_mount)

    assert job.overlaps(str(mounted / "Album"))


def test_a_claim_the_job_stopped_waiting_for_is_no_error(web) -> None:
    async def test(_client: TestClient, manager: JobManager) -> None:
        job = jobs.Job(jobs.KINDS["claim"], Folder(""), {}, "a job", None, False, False)
        claimed: concurrent.futures.Future[str | None] = concurrent.futures.Future()
        claimed.cancel()

        manager._claim(job, str(web.dest), claimed, True)  # On the server's loop: an error would be the server's

    _with_app(test)
