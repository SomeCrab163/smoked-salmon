"""salmon web never replaces a folder another running job works on (#1056).

A replace claims its destination, and is refused when another running job's folder is it, in it or holds it: the
upload's rename, its later trackers' folders and a cross-upload's. They guard deletion of users' data.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from aiohttp.test_utils import TestClient
from helpers import uploader_dry_run
from helpers.album_files import album
from helpers.cross_upload import FOLDER, TORRENT_ID, _album, cross_upload_world
from helpers.tracker_dirs import contents, tracker_dirs
from helpers.uploader_dry_run import RENAMED, fake_upload_world
from helpers.webui_jobs import AUTH, _finished, _until, _with_app
from helpers.webui_upload import PARAMS, _drive, _web_run
from test_cross_upload import (
    _source_has,
    settings,  # noqa: F401 (a fixture)
)
from test_uploader_dry_run import image_uploads  # noqa: F401 (a fixture)

from salmon import cfg, interaction
from salmon.placement import PlacedFolders
from salmon.trackers import base
from salmon.webui import jobs
from salmon.webui.jobs import JobKind, JobManager

pytestmark = pytest.mark.usefixtures("settings")


class Folder(msgspec.Struct):
    folder: str


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    """RED's and OPS's directories, and job kinds that hold a folder, or place an album for RED."""
    found = tracker_dirs(monkeypatch, tmp_path)
    found.torrents = tmp_path / "torrents"
    found.torrents.mkdir()
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(found.torrents))
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", False)
    monkeypatch.setattr(base, "_learned_secrets", set())

    async def hold(_params: Folder) -> None:
        await asyncio.Event().wait()  # Until it is cancelled

    async def claim_then_hold(params: Folder) -> None:
        await jobs.claim_folder(params.folder)
        await asyncio.Event().wait()

    async def place(params: Folder) -> str:
        return await PlacedFolders().place(params.folder, "RED")

    for name, run, held in (("hold", hold, True), ("claim", claim_then_hold, False), ("place", place, False)):
        folder = (lambda params: params.folder) if held else (lambda _params: None)
        kind = JobKind(name=name, params=Folder, run=run, title=lambda _params: "the other job", folder=folder)
        monkeypatch.setitem(jobs.KINDS, name, kind)
    return found


async def _job(client: TestClient, kind: str, params: dict[str, Any], **body: Any) -> str:
    response = await client.post("/api/jobs", json={"kind": kind, "params": params, **body}, headers=AUTH)
    assert response.status == 201, await response.text()
    return (await response.json())["id"]


async def _holding(client: TestClient, manager: JobManager, folder: Path, kind: str = "hold") -> str:
    job_id = await _job(client, kind, {"folder": str(folder)})
    job = manager.jobs
    await _until(lambda: job[job_id].status == "running" and (kind == "hold" or bool(job[job_id]._claimed)))
    return job_id


async def _stop(manager: JobManager, job_id: str) -> None:
    manager.cancel(job_id)
    await _finished(manager, job_id)


def _agree(question: dict[str, Any]) -> Any:
    return True if "replace it?" in str(question) else ""


def _asked_to_replace(asked: list[dict[str, Any]]) -> bool:
    return any("replace it?" in str(question) for question in asked)


# --- What overlaps -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hold", "claim"], ids=["started on it", "claimed it"])
@pytest.mark.parametrize("held", [".", "CD1", ".."], ids=["the destination", "inside it", "holding it"])
def test_a_folder_another_running_job_works_on_is_never_replaced(dirs, kind: str, held: str) -> None:
    source = album(dirs.downloads / "Album")
    dest = album(dirs.red / "Album")
    album(dest / "CD1")
    before = contents(dest)

    async def test(client: TestClient, manager: JobManager) -> None:
        other = await _holding(client, manager, (dest / held).resolve(), kind)
        job, asked = await _drive(client, manager, await _job(client, "place", {"folder": str(source)}), _agree)
        await _stop(manager, other)

        assert job.status == "failed"
        assert "the other job" in str(job.error)
        assert not _asked_to_replace(asked)  # Refused before the question

    _with_app(test)
    assert contents(dest) == before


def test_a_job_beside_it_is_no_reason_to_refuse(dirs) -> None:
    source = album(dirs.downloads / "Album")
    beside = album(dirs.red / "Album 2")

    async def test(client: TestClient, manager: JobManager) -> None:
        other = await _holding(client, manager, beside)
        job, _asked = await _drive(client, manager, await _job(client, "place", {"folder": str(source)}))
        await _stop(manager, other)

        assert job.status == "done", job.error

    _with_app(test)
    assert (dirs.red / "Album" / "01.flac").is_file()


@pytest.mark.parametrize("waiting_on", ["replace", "copy"])
def test_a_job_started_inside_it_while_a_question_waits_is_seen(monkeypatch, dirs, waiting_on: str) -> None:
    source = album(dirs.downloads / "Album")
    dest = album(dirs.red / "Album")
    album(dest / "CD1")
    before = contents(dest)
    monkeypatch.setattr(cfg.directory, "hardlinks", waiting_on == "copy")
    if waiting_on == "copy":

        def no_link(*_args: Any) -> None:
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr("os.link", no_link)

    async def test(client: TestClient, manager: JobManager) -> None:
        placing = await _job(client, "place", {"folder": str(source)})
        others: list[str] = []
        wanted = "replace it?" if waiting_on == "replace" else "Copy instead"
        seen: list[str] = []
        while not others:
            await _until(lambda: manager.jobs[placing].question is not None)
            question = manager.jobs[placing].question
            assert question is not None
            seen.append(str(question))
            if wanted in seen[-1]:
                others.append(await _holding(client, manager, dest / "CD1"))
            answer = {"question_id": question["id"], "value": True}
            await client.post(f"/api/jobs/{placing}/answer", json=answer, headers=AUTH)
            await _until(lambda answered=question: manager.jobs[placing].question != answered)
        job = await _finished(manager, placing)
        await _stop(manager, others[0])

        assert job.status == "failed"

    _with_app(test)
    assert contents(dest) == before


def test_the_destination_is_held_until_the_job_ends(monkeypatch, dirs) -> None:
    source = album(dirs.downloads / "Album")

    async def place_then_wait(params: Folder) -> None:
        await PlacedFolders().place(params.folder, "RED")
        await interaction.confirm("Placed. Go on?")

    monkeypatch.setitem(
        jobs.KINDS,
        "place",
        JobKind(name="place", params=Folder, run=place_then_wait, title=lambda _params: "placing"),
    )

    async def test(client: TestClient, manager: JobManager) -> None:
        placing = await _job(client, "place", {"folder": str(source)})
        await _until(lambda: manager.jobs[placing].question is not None)
        waiting = await _job(client, "hold", {"folder": str(dirs.red / "Album")})
        assert manager.jobs[waiting].status == "queued"  # As on any folder a job works on
        await _stop(manager, waiting)
        await _stop(manager, placing)

    _with_app(test)


# --- The three replace sites -------------------------------------------------------------------------------------


@pytest.mark.parametrize("site", ["the first tracker's rename", "a later tracker's folder"])
def test_an_upload_job_never_replaces_a_folder_another_job_works_on(
    monkeypatch,
    dirs,
    image_uploads,  # noqa: F811
    site: str,
) -> None:
    source = uploader_dry_run._album(dirs.downloads / "Album")
    dest = album((dirs.red if "rename" in site else dirs.ops) / RENAMED)
    album(dest / "CD1")
    before = contents(dest)
    tracker, _queued, _logins = fake_upload_world(monkeypatch, dirs.torrents)

    async def test(client: TestClient, manager: JobManager) -> None:
        other = await _holding(client, manager, dest / "CD1")
        params = {**PARAMS, "path": str(source), "trackers": ["RED", "OPS"], "conversions": []}
        job, asked = await _drive(client, manager, await _job(client, "upload", params), _agree)
        await _stop(manager, other)

        assert job.status == "failed", job.error
        assert not _asked_to_replace(asked)
        uploads = [sent for sent in tracker.sent if sent.query.get("action") == "upload"]
        assert len(uploads) == (0 if "rename" in site else 1)  # D7: abort, or skip that tracker

    _web_run(tracker, test)
    assert contents(dest) == before


def test_a_cross_upload_job_never_replaces_a_folder_holding_another_jobs(monkeypatch, dirs) -> None:
    source = _album(dirs.ops / FOLDER)
    dest = album(dirs.red / FOLDER)
    album(dest / "CD1")
    before = contents(dest)
    world = cross_upload_world(monkeypatch, dirs)

    async def test(client: TestClient, manager: JobManager) -> None:
        async with world.serving():
            _source_has(source)(world.trackers["OPS"], world.trackers["RED"])
            other = await _holding(client, manager, dest / "CD1")
            params = {"inputs": [str(TORRENT_ID)], "source": "OPS", "target": "RED"}
            job, asked = await _drive(client, manager, await _job(client, "cross_upload", params), _agree)
            await _stop(manager, other)

            assert job.status == "failed", job.error
            assert not _asked_to_replace(asked)
            assert world.trackers["RED"].posts() == []

    _with_app(test)
    assert contents(dest) == before
