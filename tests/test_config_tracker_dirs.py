"""[tracker.<code>] download_directory and the rules it loads by (#1056)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from salmon.config.validations import Cfg


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = SimpleNamespace(
        root=tmp_path,
        downloads=tmp_path / "downloads",
        torrents=tmp_path / "torrents",
        library=tmp_path / "library",
        tmp=tmp_path / "tmp",
        home=tmp_path / "home",
    )
    for folder in list(vars(found).values())[1:]:
        folder.mkdir()
    monkeypatch.setenv("HOME", str(found.home))
    monkeypatch.setenv("USERPROFILE", str(found.home))
    return found


def _load(
    dirs: SimpleNamespace,
    *,
    red: object = None,
    ops: object = None,
    dic: object = None,
    seedboxes: list[dict[str, Any]] | None = None,
    **directory: Any,
) -> Cfg:
    """The config as msgspec loads it, RED, OPS and DIC configured, each with the folder given for it, if any."""
    trackers = {}
    for code, folder in (("red", red), ("ops", ops), ("dic", dic)):
        trackers[code] = {"session": f"made-up-{code}"}
        if folder is not None:
            trackers[code]["download_directory"] = str(folder)
    raw = {
        "directory": {"download_directory": str(dirs.downloads), "dottorrents_dir": str(dirs.torrents), **directory},
        "tracker": trackers,
        "seedbox": seedboxes or [],
    }
    return msgspec.convert(raw, type=Cfg)


# --- The folder each tracker gets (T10) ------------------------------------------------------------------------------


def test_a_trackers_folder_else_the_global_one(dirs) -> None:
    config = _load(dirs, red=dirs.root / "red")

    assert config.download_directory_for("RED") == config.download_directory_for("red") == str(dirs.root / "red")
    assert config.download_directory_for("OPS") == config.download_directory_for(None) == str(dirs.downloads)
    assert config.per_tracker_directories()
    assert config.tracker_directories() == [str(dirs.root / "red")]
    assert config.data_directories() == [str(dirs.downloads), str(dirs.root / "red")]


@pytest.mark.parametrize("value", [None, ""], ids=["not set", "empty"])
def test_unset_or_empty_is_the_global_folder(dirs, value) -> None:
    config = _load(dirs, red=value)

    assert config.download_directory_for("RED") == str(dirs.downloads)
    assert not config.per_tracker_directories()
    assert config.tracker_directories() == []
    assert config.data_directories() == [str(dirs.downloads)]


def test_a_tilde_is_expanded_at_load(dirs) -> None:
    assert Path(_load(dirs, red="~/red").download_directory_for("RED")) == dirs.home / "red"


# --- What is refused (T11, T12, T14) ---------------------------------------------------------------------------------


def test_a_folder_not_there_yet_is_accepted_and_not_made(dirs) -> None:
    _load(dirs, red=dirs.root / "not" / "yet")

    assert not (dirs.root / "not").exists()


def test_a_file_is_refused(dirs) -> None:
    (dirs.root / "red").write_text("a file")

    with pytest.raises(msgspec.ValidationError, match="tracker.red.download_directory is not a valid directory"):
        _load(dirs, red=dirs.root / "red")


@pytest.mark.parametrize("red", ["library", "library/red", "library/not/yet", "linked", "linked/red"])
def test_a_folder_in_a_library_is_refused(dirs, red) -> None:
    (dirs.root / "linked").symlink_to(dirs.library, target_is_directory=True)

    with pytest.raises(msgspec.ValidationError, match="must not contain tracker.red.download_directory"):
        _load(dirs, red=dirs.root / red, library_dirs=[str(dirs.library)])


def test_a_folder_holding_a_library_is_refused(dirs) -> None:
    (dirs.root / "music" / "library").mkdir(parents=True)

    with pytest.raises(msgspec.ValidationError, match="must not be inside tracker.red.download_directory"):
        _load(dirs, red=dirs.root / "music", library_dirs=[str(dirs.root / "music" / "library")])


@pytest.mark.parametrize("red", ["downloads/red", "downloads", "."], ids=["in it", "the same", "holding it"])
def test_a_folder_is_never_compared_with_the_global_one(dirs, red) -> None:
    assert _load(dirs, red=dirs.root / red).per_tracker_directories()


def test_two_trackers_folders_may_be_nested(dirs) -> None:
    assert _load(dirs, red=dirs.root / "data", ops=dirs.root / "data" / "ops").per_tracker_directories()


@pytest.mark.parametrize("red", ["tmp", "tmp/red", "linked/red"])
def test_a_folder_clean_tmp_dir_would_delete_is_refused(dirs, red) -> None:
    (dirs.root / "linked").symlink_to(dirs.tmp, target_is_directory=True)

    with pytest.raises(msgspec.ValidationError, match="must not be inside tmp_dir"):
        _load(dirs, red=dirs.root / red, tmp_dir=str(dirs.tmp), clean_tmp_dir=True)


def test_a_folder_in_tmp_dir_is_accepted_without_clean_tmp_dir(dirs) -> None:
    assert _load(dirs, red=dirs.tmp / "red", tmp_dir=str(dirs.tmp)).per_tracker_directories()


# --- Local seedboxes (T13) -------------------------------------------------------------------------------------------

PASSWORDS = {"torrent_client": "qbittorrent+http://user:clientsecret@127.0.0.1:8080", "url": "remote,pass=urlsecret"}


@pytest.mark.parametrize(("name", "shown"), [("nas", "seedbox 'nas'"), ("", "seedbox number 2")])
def test_one_local_directory_for_different_folders_is_refused_without_a_secret(dirs, name, shown) -> None:
    boxes = [{"name": "fine"}, {"name": name, "directory": "/client/path", **PASSWORDS}]

    with pytest.raises(msgspec.ValidationError) as refused:
        _load(dirs, red=dirs.root / "red", seedboxes=boxes)

    message = str(refused.value)
    assert shown in message
    for expected in (f"RED ({dirs.root / 'red'})", f"OPS ({dirs.downloads})", "Remove its directory", "trackers"):
        assert expected in message
    assert "clientsecret" not in message and "urlsecret" not in message


def test_only_the_trackers_the_seedbox_serves_are_named(dirs) -> None:
    box = {"name": "nas", "directory": "/client", "trackers": ["red", "ops"]}

    with pytest.raises(msgspec.ValidationError) as refused:
        _load(dirs, red=dirs.root / "red", seedboxes=[box])

    assert "DIC" not in str(refused.value)


@pytest.mark.parametrize(
    "box",
    [
        {"directory": "/client", "enabled": False},
        {"directory": "/remote", "type": "rclone"},
        {"directory": ""},
        {"directory": "/client", "trackers": ["RED"]},
        {"directory": "/client", "trackers": ["OPS", "DIC"]},
    ],
    ids=["disabled", "rclone", "no directory", "one tracker", "trackers with one folder"],
)
def test_a_seedbox_without_the_conflict_is_accepted(dirs, box) -> None:
    assert _load(dirs, red=dirs.root / "red", seedboxes=[box]).seedbox


def test_two_spellings_of_one_folder_do_not_conflict(dirs) -> None:
    (dirs.root / "red").mkdir()
    (dirs.root / "also red").symlink_to(dirs.root / "red", target_is_directory=True)

    assert _load(
        dirs,
        red=dirs.root / "red",
        ops=dirs.root / "also red",
        dic=f"{dirs.root / 'red'}/",
        seedboxes=[{"directory": "/client"}],
    ).seedbox


@pytest.mark.parametrize("inner_first", [True, False])
def test_one_folder_inside_the_other_conflicts(dirs, inner_first) -> None:
    inner, outer = dirs.downloads / "red", dirs.downloads
    red, ops = (inner, outer) if inner_first else (outer, inner)

    with pytest.raises(msgspec.ValidationError, match="differs"):
        _load(dirs, red=red, ops=ops, seedboxes=[{"directory": "/client", "trackers": ["RED", "OPS"]}])


def test_without_a_key_a_local_directory_serves_every_tracker(dirs) -> None:
    assert _load(dirs, seedboxes=[{"directory": "/client"}]).seedbox


def test_a_seedbox_pinned_to_a_tracker_not_configured_is_accepted(dirs) -> None:
    raw = {
        "directory": {"download_directory": str(dirs.downloads), "dottorrents_dir": str(dirs.torrents)},
        "tracker": {"red": {"session": "made-up", "download_directory": str(dirs.root / "red")}},
        "seedbox": [{"directory": "/client", "trackers": ["DIC"]}],
    }

    assert msgspec.convert(raw, type=Cfg).seedbox
