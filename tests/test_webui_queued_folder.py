"""A salmon web job that waited runs only on the folder it was started on, never one put in its place (#1056).

Else a queued upload uploads the album that replaced its own, unattended with -yyy.
"""

import asyncio
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album
from helpers.tracker_dirs import tracker_dirs
from helpers.webui_jobs import AUTH, _finished, _question, _until, _with_app

from salmon import cfg
from salmon.placement import PlacedFolders
from salmon.trackers import base
from salmon.webui import jobs
from salmon.webui.jobs import JobKind, JobManager


class Folder(msgspec.Struct):
    folder: str


@pytest.fixture
def world(monkeypatch, tmp_path) -> SimpleNamespace:
    """An album in RED's directory, a job kind that notes what it finds in its folder, and one that takes the slot."""
    found = tracker_dirs(monkeypatch, tmp_path)
    torrents = tmp_path / "torrents"
    torrents.mkdir()
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(torrents))
    monkeypatch.setattr(base, "_learned_secrets", set())
    found.album = album(found.red / "Album")
    (found.album / "B.txt").write_text("B")
    found.other = album(found.downloads / "Album")
    (found.other / "A.txt").write_text("A")
    found.ran, found.free = [], threading.Event()

    async def work(params: Folder) -> None:
        found.ran.append(sorted(file.name for file in Path(params.folder).iterdir()))

    async def busy(_params: Folder) -> None:
        while not found.free.is_set():
            await asyncio.sleep(0.01)

    async def place(params: Folder) -> str:
        return await PlacedFolders().place(params.folder, "RED")

    for name, run, held in (("work", work, True), ("busy", busy, False), ("place", place, True)):
        folder = (lambda params: params.folder) if held else (lambda _params: None)
        kind = JobKind(name=name, params=Folder, run=run, title=lambda _params: "a job", folder=folder)
        monkeypatch.setitem(jobs.KINDS, name, kind)
    return found


@pytest.mark.parametrize("started_on", [".", "CD1"])
def test_a_queued_job_never_works_on_the_album_that_replaced_its_own(world, started_on: str) -> None:
    for folder in (world.album, world.other):
        album(folder / "CD1")

    async def test(client: TestClient, manager: JobManager) -> None:
        placing = manager.start("place", {"folder": str(world.other)})
        question = await _question(manager, placing.id)
        waiting = manager.start("work", {"folder": str(world.album / started_on)}, assume_defaults=True)
        assert waiting.status == "queued"
        answer = {"question_id": question["id"], "value": True}
        assert (await client.post(f"/api/jobs/{placing.id}/answer", json=answer, headers=AUTH)).status == 200

        assert (await _finished(manager, placing.id)).status == "done"
        job = await _finished(manager, waiting.id)
        assert job.status == "failed"
        assert "start it again" in str(job.error)

    _with_app(test, max_jobs=1)
    assert (world.album / "A.txt").exists()
    assert world.ran == []


def _remade(folder: Path) -> None:
    # As a re-transcode: on ext4 the folder made has the inode of the one deleted
    shutil.rmtree(folder)
    folder.mkdir()


def _swapped(folder: Path) -> None:
    folder.rename(folder.with_name("moved away"))
    album(folder)


def _gone(folder: Path) -> None:
    shutil.rmtree(folder)


def _changed_in_place(folder: Path) -> None:
    (folder / "B.txt").write_text("retagged")
    (folder / "02.flac").rename(folder / "02 - two.flac")


def _while_it_waits(world: SimpleNamespace, change: Callable[[Path], None]) -> str:
    """Change the album while a job on it waits for the one slot; that job's status once it ended."""
    ended: list[str] = []

    async def test(_client: TestClient, manager: JobManager) -> None:
        manager.start("busy", {"folder": ""})
        waiting = manager.start("work", {"folder": str(world.album)})
        assert waiting.status == "queued"
        change(world.album)
        world.free.set()
        ended.append((await _finished(manager, waiting.id)).status)

    _with_app(test, max_jobs=1)
    return ended[0]


@pytest.mark.parametrize("change", [_remade, _swapped, _gone])
def test_a_queued_job_fails_when_its_folder_is_another(world, change: Callable[[Path], None]) -> None:
    assert _while_it_waits(world, change) == "failed"
    assert world.ran == []


def test_a_queued_job_runs_on_its_folder_changed_in_place(world) -> None:
    assert _while_it_waits(world, _changed_in_place) == "done"  # As after the tag job it waited for
    assert len(world.ran) == 1


def test_with_no_key_a_queued_job_runs_on_whatever_is_there(monkeypatch, world) -> None:
    for tracker in (cfg.tracker.red, cfg.tracker.ops):
        monkeypatch.setattr(tracker, "download_directory", None)

    assert _while_it_waits(world, _swapped) == "done"


def _open_on(folder: Path) -> int:
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            count += os.readlink(f"/proc/self/fd/{fd}") == str(folder.resolve())
        except OSError:  # The listing's own, closed since
            continue
    return count


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="Linux's list of open files")
@pytest.mark.parametrize("end", ["ran", "cancelled while it waited"])
def test_a_job_lets_go_of_its_folder_when_it_ends(world, end: str) -> None:
    async def test(_client: TestClient, manager: JobManager) -> None:
        manager.start("busy", {"folder": ""})
        waiting = manager.start("work", {"folder": str(world.album)})
        assert _open_on(world.album) == 1
        if end != "ran":
            manager.cancel(waiting.id)
        world.free.set()
        await _finished(manager, waiting.id)
        await _until(lambda: _open_on(world.album) == 0)

    _with_app(test, max_jobs=1)
