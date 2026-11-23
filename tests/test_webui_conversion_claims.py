"""In salmon web a converter claims the folder it writes (#1056): no other job replaces it, and it writes into none.

Through the real converters, their encoders faked. They guard deletion of users' data.
"""

import asyncio
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import msgspec
import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album, write_flac
from helpers.tracker_dirs import contents, tracker_dirs
from helpers.webui_jobs import AUTH, _finished, _until, _with_app
from helpers.webui_upload import _drive

from salmon import cfg, dryrun, interaction, placement
from salmon.converter import downconverting, transcoding
from salmon.placement import PlacedFolders
from salmon.trackers import base
from salmon.webui import jobs
from salmon.webui.jobs import JobKind, JobManager

CONVERTERS = ("transcode", "downconvert")


class Folder(msgspec.Struct):
    folder: str


@pytest.fixture
def world(monkeypatch, tmp_path) -> SimpleNamespace:
    """An album in RED's directory, where its conversions go too, and encoders that wait until they are let go."""
    found = tracker_dirs(monkeypatch, tmp_path)
    torrents = tmp_path / "torrents"
    torrents.mkdir()
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(torrents))
    monkeypatch.setattr(cfg, "seedbox", [])
    monkeypatch.setattr(base, "_learned_secrets", set())
    found.album = album(found.red / "Album [24bit FLAC]")
    for track in found.album.glob("*.flac"):
        write_flac(track, bits=24, rate=44100, title=track.stem)
    found.outputs = {"transcode": found.red / "Album [MP3 V0]", "downconvert": found.red / "Album [FLAC]"}
    found.encoding, found.go_on = threading.Event(), threading.Event()
    found.go_on.set()

    async def encode(items: list[Any], _kind: Any) -> None:
        found.encoding.set()
        while not found.go_on.is_set():
            await asyncio.sleep(0.01)
        for item in items:
            Path(item.dst).parent.mkdir(parents=True, exist_ok=True)
            if item.dst.endswith(".flac"):
                write_flac(Path(item.dst), bits=16, rate=44100)
            else:
                Path(item.dst).write_bytes(b"mp3")

    monkeypatch.setattr(transcoding, "_transcode_audio_files", encode)
    monkeypatch.setattr(downconverting, "_convert_audio_files", encode)

    async def hold(_params: Folder) -> None:
        await asyncio.Event().wait()

    async def place(params: Folder) -> str:
        return await PlacedFolders().place(params.folder, "RED")

    async def convert_then_go_on(params: Folder) -> None:
        # As an upload job: it makes the conversion, then uploads and seeds it.
        await _convert(params.folder, found.album)
        await interaction.confirm("Made. Go on?")

    for name, run, held in (("hold", hold, True), ("place", place, False), ("make", convert_then_go_on, False)):
        folder = (lambda params: params.folder) if held else (lambda _params: None)
        kind = JobKind(name=name, params=Folder, run=run, title=lambda _params: "the other job", folder=folder)
        monkeypatch.setitem(jobs.KINDS, name, kind)
    return found


async def _convert(converter: str, source: Path) -> None:
    if converter == "transcode":
        await transcoding.transcode_folder(str(source), "V0")
    else:
        await downconverting.convert_folder(str(source))


async def _start(client: TestClient, kind: str, params: dict[str, Any]) -> str:
    response = await client.post("/api/jobs", json={"kind": kind, "params": params}, headers=AUTH)
    assert response.status == 201, await response.text()
    return (await response.json())["id"]


async def _start_converter(client: TestClient, converter: str, source: Path) -> str:
    params = {"path": str(source), **({"bitrate": "V0"} if converter == "transcode" else {})}
    return await _start(client, converter, params)


async def _stop(manager: JobManager, job_id: str) -> None:
    manager.cancel(job_id)
    await _finished(manager, job_id)


def _agree(question: dict[str, Any]) -> Any:
    return True if "replace it?" in str(question) or "Delete it" in str(question) else ""


# --- What a converter writes is not replaced -----------------------------------------------------------------------


@pytest.mark.parametrize("converter", CONVERTERS)
def test_a_conversion_a_running_job_made_is_never_replaced(world, converter: str) -> None:
    output = world.outputs[converter]
    other = album(world.downloads / output.name, tags={"album": "another"})
    after_making: dict[str, str] = {}

    async def test(client: TestClient, manager: JobManager) -> None:
        making = await _start(client, "make", {"folder": converter})
        await _until(lambda: manager.jobs[making].question is not None)
        after_making.update(contents(output))
        job, asked = await _drive(client, manager, await _start(client, "place", {"folder": str(other)}), _agree)
        await _stop(manager, making)

        assert job.status == "failed"
        assert asked == []

    _with_app(test)
    assert contents(output) == after_making


@pytest.mark.parametrize("converter", CONVERTERS)
def test_a_convert_jobs_output_is_held_while_it_encodes(world, converter: str) -> None:
    output = world.outputs[converter]
    other = album(world.downloads / output.name, tags={"album": "another"})
    world.go_on.clear()

    async def test(client: TestClient, manager: JobManager) -> None:
        converting = await _start_converter(client, converter, world.album)
        await _until(world.encoding.is_set)
        job, asked = await _drive(client, manager, await _start(client, "place", {"folder": str(other)}), _agree)
        world.go_on.set()
        converted = await _finished(manager, converting)

        assert job.status == "failed"
        assert asked == []
        assert converted.status == "done", converted.error

    _with_app(test)
    expected = {".mp3"} if converter == "transcode" else {".flac"}
    assert {Path(name).suffix for name in contents(output)} == expected
    assert contents(output) != contents(other)


# --- A converter writes into no other job's folder -----------------------------------------------------------------


@pytest.mark.parametrize("converter", CONVERTERS)
@pytest.mark.parametrize("held", [".", "CD1"], ids=["the output", "inside it"])
def test_a_converter_refuses_an_output_another_job_works_on(world, converter: str, held: str) -> None:
    output = album(world.outputs[converter])
    album(output / "CD1")
    before = contents(output)

    async def test(client: TestClient, manager: JobManager) -> None:
        holding = await _start(client, "hold", {"folder": str((output / held).resolve())})
        await _until(lambda: manager.jobs[holding].status == "running")
        job, asked = await _drive(client, manager, await _start_converter(client, converter, world.album), _agree)
        await _stop(manager, holding)

        assert job.status == "failed"
        assert "the other job" in str(job.error)
        assert asked == []

    _with_app(test)
    assert contents(output) == before


def test_a_retranscode_sees_a_job_started_inside_while_its_question_waits(world) -> None:
    output = album(world.outputs["transcode"])
    album(output / "CD1")
    before = contents(output)

    async def test(client: TestClient, manager: JobManager) -> None:
        converting = await _start_converter(client, "transcode", world.album)
        await _until(lambda: manager.jobs[converting].question is not None)
        holding = await _start(client, "hold", {"folder": str(output / "CD1")})
        await _until(lambda: manager.jobs[holding].status == "running")
        job, asked = await _drive(client, manager, converting, _agree)
        await _stop(manager, holding)

        assert "Delete it" in str(asked)
        assert job.status == "failed"

    _with_app(test)
    assert contents(output) == before


# --- Only with a key, and never in a dry run -----------------------------------------------------------------------


@pytest.mark.parametrize("why", ["a key", "no key", "dry run"])
@pytest.mark.parametrize("converter", CONVERTERS)
def test_an_output_is_claimed_only_with_a_key_and_outside_a_dry_run(
    monkeypatch, world, converter: str, why: str
) -> None:
    if why == "no key":
        monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
        monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)
    claimed: list[Path] = []

    async def claim(folder: str) -> None:
        claimed.append(Path(folder))

    with placement.claiming(claim), dryrun.mode(why == "dry run"):
        anyio.run(partial(_convert, converter, world.album))

    assert set(claimed) == ({world.outputs[converter]} if why == "a key" else set())
