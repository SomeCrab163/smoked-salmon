"""salmon web with per-tracker directories (#1056): never an album themselves."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album
from helpers.tracker_dirs import tracker_dirs
from helpers.webui_jobs import AUTH, _with_app

from salmon import cfg
from salmon.trackers import base
from salmon.webui import paths
from salmon.webui.jobs import JobManager

GUARD = "Refusing a tracker's download_directory, or a folder that holds one: pick one album folder in it."


@pytest.fixture(autouse=True)
def no_learned_secrets(monkeypatch) -> None:
    """Forget what other tests' fake trackers gave, short values among them: they would mask words in a reply."""
    monkeypatch.setattr(base, "_learned_secrets", set())


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    """RED's directory inside download_directory, full of albums, which has the shape of a box set."""
    found = tracker_dirs(monkeypatch, tmp_path, "RED")
    found.red = found.downloads / "red"
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(found.red))
    monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(tmp_path / "torrents"))
    monkeypatch.setattr(cfg.directory, "cross_upload_dottorrents_dir", None)
    monkeypatch.setattr(cfg, "seedbox", [])
    (tmp_path / "torrents").mkdir()
    return found


def _refused(path: Path) -> str:
    with pytest.raises(paths.PathRefused) as refused:
        paths.album_folder(str(path))
    assert refused.value.status == 403
    return refused.value.detail


def _offered(listed: Path) -> dict[str, bool]:
    return {folder["name"]: folder["audio"] for folder in paths.listing(str(listed))["folders"]}


# --- Never an album (T29) --------------------------------------------------------------------------------------------


def test_a_trackers_directory_full_of_albums_is_refused_and_not_offered(dirs) -> None:
    album(dirs.red / "Album")

    assert _refused(dirs.red) == GUARD
    assert _offered(dirs.downloads)["red"] is False
    assert paths.listing(str(dirs.red))["audio"] is False


@pytest.mark.parametrize("made", [True, False], ids=["there", "not made yet"])
def test_a_folder_holding_a_trackers_directory_is_refused_and_not_offered(monkeypatch, dirs, made) -> None:
    holder = album(dirs.downloads / "trackers")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(holder / "red"))
    if made:
        (holder / "red").mkdir()

    assert _refused(holder) == GUARD
    assert _offered(dirs.downloads)["trackers"] is False
    assert paths.listing(str(holder))["refused"] == GUARD


def test_a_directory_spelled_through_a_link_is_refused(monkeypatch, dirs) -> None:
    holder = album(dirs.downloads / "trackers")
    (holder / "red").mkdir()
    (dirs.root / "data").symlink_to(holder, target_is_directory=True)
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(dirs.root / "data" / "red"))

    assert _refused(holder) == GUARD
    assert _offered(dirs.downloads)["trackers"] is False


def test_a_second_mount_of_a_folder_holding_one_is_refused(monkeypatch, dirs) -> None:
    holder = dirs.root / "data"
    (holder / "red").mkdir(parents=True)
    mounted = album(dirs.downloads / "mounted")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(holder / "red"))
    real_stat = paths.os.stat

    def stat(path, *args, **kwargs):
        # As a second mount of holder: os.stat says they are one folder.
        return real_stat(str(holder) if str(path) == str(mounted) else path, *args, **kwargs)

    monkeypatch.setattr(paths.os, "stat", stat)

    assert _refused(mounted) == GUARD
    assert _offered(dirs.downloads)["mounted"] is False


def test_an_album_in_a_trackers_directory_is_accepted_and_offered(dirs) -> None:
    inside = album(dirs.red / "Album")

    assert paths.album_folder(str(inside)) == str(inside)
    assert _offered(dirs.red)["Album"] is True


def test_with_no_key_the_same_folder_is_an_album_as_before(monkeypatch, dirs) -> None:
    album(dirs.red / "Album")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)

    assert paths.album_folder(str(dirs.red)) == str(dirs.red)
    assert _offered(dirs.downloads)["red"] is True


def test_a_job_on_a_folder_holding_one_is_refused_before_it_starts(monkeypatch, dirs) -> None:
    holder = album(dirs.downloads / "trackers")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(holder / "red"))

    async def test(client: TestClient, manager: JobManager) -> None:
        body = {"kind": "tag", "params": {"path": str(holder), "source": "WEB"}}
        response = await client.post("/api/jobs", json=body, headers=AUTH)

        assert response.status == 403
        assert (await response.json())["detail"] == GUARD
        assert manager.jobs == {}

    _with_app(test)
