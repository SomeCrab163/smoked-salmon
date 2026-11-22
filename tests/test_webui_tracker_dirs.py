"""salmon web with per-tracker directories (#1056): browsable roots, never an album, that open nothing around them."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient
from helpers.album_files import album
from helpers.tracker_dirs import tracker_dirs
from helpers.webui_jobs import AUTH, _with_app

from salmon import cfg
from salmon.config.validations import Seedbox
from salmon.trackers import base
from salmon.webui import jobs, kinds, paths
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


def test_a_trackers_directory_full_of_albums_is_refused_and_not_offered(monkeypatch, dirs) -> None:
    album(dirs.red / "Album")

    assert _refused(dirs.red) == "Refusing a root folder: pick one album folder in it."
    with monkeypatch.context() as without_roots:
        # The guard holds by itself, whether or not the directory is a root.
        without_roots.setattr(paths, "roots", paths._fixed_roots)
        assert _refused(dirs.red) == GUARD
        assert _offered(dirs.downloads)["red"] is False
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


# --- The roots (T30) -------------------------------------------------------------------------------------------------


@pytest.fixture
def outside(monkeypatch, dirs) -> SimpleNamespace:
    """RED's directory outside download_directory, with an album, and folders beside and above it."""
    dirs.red = dirs.root / "data" / "red"
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(dirs.red))
    album(dirs.red / "Album")
    album(dirs.root / "data" / "Sibling")
    return dirs


def test_a_directory_outside_the_global_one_is_a_browsable_root(outside) -> None:
    assert [root.path for root in paths.roots()] == [str(outside.downloads), str(outside.red)]
    assert paths.browsable_folder(str(outside.red)) == str(outside.red)
    assert _offered(outside.red) == {"Album": True}
    assert paths.listing(str(outside.red))["parent"] is None


def test_its_albums_work_for_a_job_and_as_a_cross_uploads_path(outside) -> None:
    given = str(outside.red / "Album")

    tag = jobs.KINDS["tag"].check(kinds.TagParams(path=given, source="WEB"), False)
    cross = jobs.KINDS["cross_upload"].check(
        kinds.CrossUploadParams(inputs=["123"], source="RED", target="OPS", path=given), False
    )

    assert (tag.path, cross.path) == (given, given)


def test_a_directory_not_there_yet_is_not_listed(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(dirs.root / "not yet"))

    assert [root.path for root in paths.roots()] == [str(dirs.downloads)]


def test_a_directory_that_is_the_global_one_is_listed_once(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", f"{dirs.downloads}/")

    assert [root.path for root in paths.roots()] == [str(dirs.downloads)]


def test_the_roots_order_is_global_trackers_libraries(monkeypatch, outside) -> None:
    monkeypatch.setattr(cfg.directory, "library_dirs", [str(outside.library)])

    assert [(root.path, root.library) for root in paths.roots()] == [
        (str(outside.downloads), False),
        (str(outside.red), False),
        (str(outside.library), True),
    ]


@pytest.mark.parametrize("folder", ["data", "data/Sibling"])
def test_its_parent_and_siblings_are_still_refused(outside, folder) -> None:
    path = outside.root / folder

    assert "outside" in _refused(path)
    with pytest.raises(paths.PathRefused):
        paths.listing(str(path))
    with pytest.raises(paths.PathRefused):
        paths.browsable_folder(str(path))


# --- It opens nothing around itself (T31) ----------------------------------------------------------------------------


def _withhold(monkeypatch, how: str, folder: Path) -> None:
    if how == "dottorrents_dir":
        monkeypatch.setattr(cfg.directory, "dottorrents_dir", str(folder))
    else:
        monkeypatch.setattr(cfg, "seedbox", [Seedbox(name="box", type="local", directory=str(folder))])


@pytest.mark.parametrize("how", ["dottorrents_dir", "seedbox"])
@pytest.mark.parametrize("album_in", ["Loose", "red/Album"], ids=["beside it", "in it"])
def test_a_directory_inside_a_withheld_folder_unlocks_nothing(monkeypatch, dirs, how, album_in) -> None:
    around = dirs.downloads / "watch"
    (around / "red").mkdir(parents=True)
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(around / "red"))
    _withhold(monkeypatch, how, around)
    inside = album(around / album_in, cover=True)

    assert "folder of .torrent files or a seedbox's" in _refused(inside)
    with pytest.raises(paths.PathRefused, match="seedbox"):
        paths.image_file(str(inside / "cover.jpg"), content=False)
    assert _offered(inside.parent)[inside.name] is False
    assert paths.listing(str(inside), images=True)["images"] == []


def test_a_withheld_folder_outside_the_global_one_stays_withheld_in_a_trackers_root(monkeypatch, outside) -> None:
    _withhold(monkeypatch, "seedbox", outside.root / "data")

    assert "seedbox" in _refused(outside.red / "Album")


def test_a_seedbox_folder_inside_a_trackers_directory_stays_withheld(monkeypatch, outside) -> None:
    inside = album(outside.red / "seed" / "Album")
    _withhold(monkeypatch, "seedbox", outside.red / "seed")

    assert "seedbox" in _refused(inside)
    assert paths.album_folder(str(outside.red / "Album")) == str(outside.red / "Album")


def test_a_directory_that_is_the_seedboxs_folder_is_usable(monkeypatch, outside) -> None:
    usable = album(outside.red / "With cover", cover=True)
    _withhold(monkeypatch, "seedbox", outside.red)

    assert paths.album_folder(str(usable)) == str(usable)
    assert paths.image_file(str(usable / "cover.jpg"), content=False) == str(usable / "cover.jpg")


def test_a_directory_that_is_dottorrents_dir_takes_albums_but_sends_no_image(monkeypatch, outside) -> None:
    usable = album(outside.red / "With cover", cover=True)
    _withhold(monkeypatch, "dottorrents_dir", outside.red)

    assert paths.album_folder(str(usable)) == str(usable)
    with pytest.raises(paths.PathRefused, match=".torrent"):
        paths.image_file(str(usable / "cover.jpg"), content=False)
