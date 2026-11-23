"""A cancelled placement replaces nothing (#1056): Ctrl-C as a real signal, and salmon web's Cancel.

The first Ctrl-C only cancels the task, which takes effect where the task next waits. They guard deletion of users'
data.
"""

import os
import shutil
import signal
import threading
from functools import partial
from types import SimpleNamespace
from typing import Any

import anyio
import msgspec
import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album, write_flac
from helpers.tracker_dirs import building_left, contents, links_fail, tracker_dirs
from helpers.uploader_dry_run import METADATA, _returning
from helpers.webui_jobs import AUTH, _finished, _question, _until, _with_app

import salmon.tagger.foldername
from salmon import cfg, interaction, placement
from salmon.converter import transcoding
from salmon.placement import PlacedFolders
from salmon.tagger.foldername import rename_folder
from salmon.trackers import base
from salmon.uploader import spectrals
from salmon.webui import jobs
from salmon.webui.jobs import JobKind, JobManager


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.album = album(found.downloads / "Album")
    found.dest = album(found.red / "Album", tags={"album": "the one there"})
    (found.dest / "only here.txt").write_text("old")
    return found


def _ctrl_c() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def _interrupting(monkeypatch, at: str, *, answered: bool = True) -> None:
    """Agree to everything, with a Ctrl-C at the question that holds `at`, which is then answered `answered`."""

    async def confirm(text: str, **_kwargs: Any) -> bool:
        if at in text:
            _ctrl_c()
            return answered
        return True

    monkeypatch.setattr(interaction, "confirm", confirm)


@pytest.mark.parametrize("answered", [True, False], ids=["yes", "no"])
@pytest.mark.parametrize("question", ["replace it?", "Copy instead?"])
def test_a_ctrl_c_at_a_question_stops_the_run_whatever_the_answer(monkeypatch, dirs, question, answered) -> None:
    before = contents(dirs.dest)
    links_fail(monkeypatch)
    _interrupting(monkeypatch, question, answered=answered)

    # Not PlacementDeclined for a no: the run would go on to the next tracker.
    with pytest.raises(KeyboardInterrupt):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    assert contents(dirs.dest) == before
    assert building_left(dirs.root) == []


def test_a_ctrl_c_at_the_copy_question_copies_nothing(monkeypatch, dirs) -> None:
    links_fail(monkeypatch)
    _interrupting(monkeypatch, "Copy instead?")
    copied: list[str] = []
    monkeypatch.setattr(shutil, "copy2", lambda src, _dst, **_kwargs: copied.append(src))

    with pytest.raises(KeyboardInterrupt):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    assert copied == []  # An album can take minutes


@pytest.mark.parametrize("linked", [True, False], ids=["link", "copy"])
def test_a_ctrl_c_while_the_folder_is_built_replaces_nothing(monkeypatch, dirs, linked: bool) -> None:
    before = contents(dirs.dest)
    monkeypatch.setattr(cfg.directory, "hardlinks", linked)
    _interrupting(monkeypatch, "no such question")
    module, name = (os, "link") if linked else (shutil, "copy2")
    real, pressed = getattr(module, name), []

    def then_ctrl_c(*args: Any, **kwargs: Any) -> Any:
        made = real(*args, **kwargs)
        if not pressed:
            pressed.append(True)
            _ctrl_c()
        return made

    monkeypatch.setattr(module, name, then_ctrl_c)

    with pytest.raises(KeyboardInterrupt):
        anyio.run(PlacedFolders().place, str(dirs.album), "RED")

    assert contents(dirs.dest) == before
    assert building_left(dirs.root) == []


def test_a_ctrl_c_once_the_folder_is_in_place_stops_the_run_before_its_upload(monkeypatch, dirs) -> None:
    _interrupting(monkeypatch, "no such question")
    remove = shutil.rmtree

    def then_ctrl_c(path: str, *args: Any, **kwargs: Any) -> None:
        remove(path, *args, **kwargs)
        if placement.REPLACED_PREFIX in str(path):
            _ctrl_c()

    monkeypatch.setattr(shutil, "rmtree", then_ctrl_c)
    uploading: list[str] = []

    async def place_then_upload() -> None:
        # As upload_and_report: nothing waits before it sends
        uploading.append(await PlacedFolders().place(str(dirs.album), "RED"))

    with pytest.raises(KeyboardInterrupt):
        anyio.run(place_then_upload)

    assert uploading == []
    assert contents(dirs.dest) == contents(dirs.album)


def test_a_ctrl_c_once_the_renamed_folder_is_in_place_still_moves_its_spectrals(monkeypatch, dirs) -> None:
    album(dirs.red / "New", tags={"album": "the one there"})
    (dirs.album / "Spectrals").mkdir()
    monkeypatch.setattr(spectrals, "made_by_salmon", lambda _path: True)
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", _returning("New"))
    moved: list[str] = []
    monkeypatch.setattr(salmon.tagger.foldername, "_move_specs_folder", lambda src, _to: moved.append(src))
    _interrupting(monkeypatch, "no such question")
    remove = shutil.rmtree

    def then_ctrl_c(path: str, *args: Any, **kwargs: Any) -> None:
        remove(path, *args, **kwargs)
        if placement.REPLACED_PREFIX in str(path):
            _ctrl_c()

    monkeypatch.setattr(shutil, "rmtree", then_ctrl_c)

    with pytest.raises(KeyboardInterrupt):
        anyio.run(partial(rename_folder, str(dirs.album), dict(METADATA), True, parent=str(dirs.red)))

    assert len(moved) == 1  # The source must not keep a spectrals folder the upload never deletes


def test_a_ctrl_c_once_the_renamed_folder_is_in_place_stops_the_rename(monkeypatch, dirs) -> None:
    album(dirs.red / "New", tags={"album": "the one there"})
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", _returning("New"))
    _interrupting(monkeypatch, "no such question")
    remove = shutil.rmtree

    def then_ctrl_c(path: str, *args: Any, **kwargs: Any) -> None:
        remove(path, *args, **kwargs)
        if placement.REPLACED_PREFIX in str(path):
            _ctrl_c()

    monkeypatch.setattr(shutil, "rmtree", then_ctrl_c)
    went_on: list[bool] = []

    async def rename_then_go_on() -> None:
        await rename_folder(str(dirs.album), dict(METADATA), True, parent=str(dirs.red))
        went_on.append(True)  # As rename_files would, before anything waits

    with pytest.raises(KeyboardInterrupt):
        anyio.run(rename_then_go_on)

    assert went_on == []
    assert contents(dirs.red / "New") == contents(dirs.album)


def test_a_ctrl_c_at_the_retranscode_question_deletes_nothing(monkeypatch, dirs) -> None:
    source = album(dirs.downloads / "Album [FLAC]")
    for track in source.glob("*.flac"):
        write_flac(track, bits=16, rate=44100, title=track.stem)
    output = album(dirs.red / "Album [MP3 V0]")
    before = contents(output)
    _interrupting(monkeypatch, "Delete it")

    async def no_encoder(*_args: Any) -> None:
        pass

    monkeypatch.setattr(transcoding, "_transcode_audio_files", no_encoder)

    with pytest.raises(KeyboardInterrupt):
        anyio.run(partial(transcoding.transcode_folder, str(source), "V0", output_dir=str(dirs.red)))

    assert contents(output) == before


class Folder(msgspec.Struct):
    folder: str


def test_a_cancel_holds_when_the_claim_is_answered_before_the_job_waits(monkeypatch, dirs, tmp_path) -> None:
    torrents = tmp_path / "torrents"
    torrents.mkdir()
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(torrents))
    monkeypatch.setattr(cfg.directory, "hardlinks", False)
    monkeypatch.setattr(base, "_learned_secrets", set())
    before = contents(dirs.dest)
    copying, go_on = threading.Event(), threading.Event()
    copytree, call = shutil.copytree, JobManager._call

    def slow_copy(*args: Any, **kwargs: Any) -> Any:
        copying.set()
        assert go_on.wait(10)
        return copytree(*args, **kwargs)

    def answered_at_once(manager: JobManager, callback: Any, *args: Any) -> None:
        call(manager, callback, *args)
        if getattr(callback, "__name__", "") == "_claim":
            args[2].result(timeout=10)  # As on a busy machine: the server's thread runs first

    async def place(params: Folder) -> str:
        return await PlacedFolders().place(params.folder, "RED")

    monkeypatch.setattr(shutil, "copytree", slow_copy)
    monkeypatch.setattr(JobManager, "_call", answered_at_once)
    monkeypatch.setitem(jobs.KINDS, "place", JobKind(name="place", params=Folder, run=place, title=lambda _: "place"))

    async def test(client: TestClient, manager: JobManager) -> None:
        job = manager.start("place", {"folder": str(dirs.album)})
        question = await _question(manager, job.id)
        answer = {"question_id": question["id"], "value": True}
        assert (await client.post(f"/api/jobs/{job.id}/answer", json=answer, headers=AUTH)).status == 200
        await _until(copying.is_set)
        manager.cancel(job.id)
        go_on.set()

        assert (await _finished(manager, job.id)).status == "cancelled"

    _with_app(test)
    assert contents(dirs.dest) == before
