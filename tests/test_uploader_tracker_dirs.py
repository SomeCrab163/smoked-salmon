"""salmon up with per-tracker directories (#1056): the first tracker's folder, the later ones', and the conversions'.

They guard deletion of users' data, so each guard has a case of its own.
"""

import importlib
import os
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import asyncclick as click
import pytest
from helpers.album_files import album, write_flac
from helpers.tracker_dirs import answer, building_left, contents, inodes, links_fail, tracker_dirs
from helpers.uploader_dry_run import METADATA, RENAMED, _album, _returning, _returning_async, _run_up, _transcode
from test_uploader_dry_run import (
    image_uploads,  # noqa: F401 (a fixture: see images)
)
from torf import Torrent

import salmon.tagger.foldername
import salmon.uploader
from salmon import cfg, interaction, placement
from salmon.config.validations import Seedbox
from salmon.converter import downconverting, transcoding
from salmon.converter.conversions import conversion_of, record_conversion
from salmon.errors import RequestFailedError, UploadError
from salmon.placement import PlacedFolders, PlacementDeclined, PlacementError, Replacing
from salmon.tagger.foldername import remove_source_folder, rename_folder
from salmon.trackers.ops import OpsApi
from salmon.trackers.red import RedApi
from salmon.uploader import conversion_upload, record
from salmon.uploader.seedbox import UploadManager
from salmon.uploader.torrent_client import QBittorrentClient

# Not `from salmon.uploader import upload`: that is the function.
upload_module = importlib.import_module("salmon.uploader.upload")

# Covers and spectrals go to the fake image host.
pytestmark = pytest.mark.usefixtures("images")


@pytest.fixture
def images(image_uploads) -> list[tuple[str, str]]:  # noqa: F811
    return image_uploads


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    found = tracker_dirs(monkeypatch, tmp_path)
    found.torrents = tmp_path / "torrents"
    found.torrents.mkdir()
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", False)
    return found


def _up(monkeypatch, dirs: SimpleNamespace, source: Path, *trackers: str, conversions: str = "none", **kwargs: Any):
    """salmon up -t `trackers`, with what it would exit with."""
    with record.recording() as recorded:
        run = _run_up(
            monkeypatch,
            source,
            dirs.torrents,
            ("--conversions", conversions, *kwargs.pop("args", ())),
            trackers=trackers,
            **kwargs,
        )
    return run, recorded.exit_status(run.result.exit_code)


def _rename(monkeypatch, path: Path, name: str, parent: Path | None, **kwargs: Any) -> str:
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", _returning(name))
    given = str(parent) if parent else None
    return anyio.run(partial(rename_folder, str(path), dict(METADATA), True, parent=given, **kwargs))


def _earlier(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "other.flac").write_bytes(b"another album")
    os.utime(folder, ns=(1_000_000_000, 1_000_000_000))
    return folder


# --- The first tracker's directory (T17) -----------------------------------------------------------------------------


def test_up_renames_into_the_first_trackers_directory_and_makes_it(monkeypatch, dirs) -> None:
    source = _album(dirs.root / "in" / "Album")

    run, status = _up(monkeypatch, dirs, source, "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert f"Created {dirs.red}." in run.result.output
    assert f"Hardlinked folder to '{dirs.red / RENAMED}'." in run.result.output
    assert len(list((dirs.red / RENAMED).glob("*.flac"))) == 2
    assert not (dirs.downloads / RENAMED).exists()


def test_a_first_tracker_without_a_key_gets_the_global_directory(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert (dirs.downloads / RENAMED).is_dir()


def test_with_no_key_up_renames_into_the_global_one_and_copies_silently(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)
    links_fail(monkeypatch)
    asked = answer(monkeypatch)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "Hardlinking didn't work, falling back to non-hardlink copy..." in run.result.output
    assert (dirs.downloads / RENAMED).is_dir()
    assert asked == []


def test_a_rename_that_cannot_link_under_yyy_copies_and_says_so(monkeypatch, dirs) -> None:
    links_fail(monkeypatch)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "-yyy: copying" in run.result.output
    assert building_left(dirs.root) == []


def test_a_declined_copy_aborts_the_rename_and_keeps_the_folder_to_replace(monkeypatch, dirs) -> None:
    source = album(dirs.root / "in" / "Album")
    before = contents(_earlier(dirs.red / "New"))
    links_fail(monkeypatch)
    answer(monkeypatch, True, False)  # Replace it; don't copy

    with pytest.raises(click.Abort):
        _rename(monkeypatch, source, "New", dirs.red, check=True)

    assert contents(dirs.red / "New") == before
    assert len(list(source.glob("*.flac"))) == 2


def test_the_renamed_folder_replaces_whole_and_never_merges(monkeypatch, dirs) -> None:
    source = album(dirs.root / "in" / "New")
    record_conversion(str(_earlier(dirs.red / "New")), source="/of/the/old", kind="transcode", bitrate="V0")
    answer(monkeypatch, True)

    assert _rename(monkeypatch, source, "New", dirs.red) == str(dirs.red / "New")

    assert contents(dirs.red / "New") == contents(source)
    assert conversion_of(str(dirs.red / "New")) is None


def test_keeping_the_folder_there_stops_the_rename(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs.red / "New"))
    asked = answer(monkeypatch, False)

    with pytest.raises(UploadError, match="already exists"):
        _rename(monkeypatch, album(dirs.root / "in" / "New"), "New", dirs.red)

    assert asked[0][1]["default"] is False
    assert contents(dirs.red / "New") == before


def test_a_folder_recreated_during_the_renames_question_is_not_replaced(monkeypatch, dirs) -> None:
    dest = _earlier(dirs.red / "New")

    async def recreate_then_agree(*_args: Any, **_kwargs: Any) -> bool:
        (dest / "other.flac").unlink()
        dest.rmdir()
        dest.mkdir()
        (dest / "new download.flac").write_bytes(b"new")
        return True

    monkeypatch.setattr(interaction, "confirm", recreate_then_agree)

    with pytest.raises(click.Abort):
        _rename(monkeypatch, album(dirs.root / "in" / "New"), "New", dirs.red)

    assert [file.name for file in dest.iterdir()] == ["new download.flac"]


def test_a_run_under_yyy_aborts_at_a_folder_already_there(monkeypatch, dirs) -> None:
    before = contents(_earlier(dirs.red / RENAMED))

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert status == 1, run.result.output
    assert "Not replacing" in run.result.output and "Aborting upload" in run.result.output
    assert "would you like to replace it?" not in run.result.output
    assert contents(dirs.red / RENAMED) == before
    assert run.tracker.not_gets() == []


@pytest.mark.parametrize("which", ["inside the album", "holding the album", "holding the original", "a link"])
def test_a_rename_never_goes_over_what_the_run_works_from(monkeypatch, dirs, which) -> None:
    original, parent = None, dirs.red
    if which == "inside the album":
        source = album(dirs.root / "box")
        parent = source / "red"
        parent.mkdir()
        monkeypatch.setattr(cfg.tracker.red, "download_directory", str(parent))
    elif which == "holding the album":
        source = album(dirs.red / "New" / "CD1")
    elif which == "holding the original":
        original = album(dirs.red / "New" / "CD1")
        source = album(dirs.root / "copy")
    else:
        source = album(dirs.root / "in")
        album(dirs.red / "real")
        (dirs.red / "New").symlink_to(dirs.red / "real", target_is_directory=True)
    before = contents(dirs.root)
    asked = answer(monkeypatch)

    with pytest.raises(click.Abort):
        _rename(monkeypatch, source, "New", parent, original=str(original) if original else None)

    assert asked == []
    assert contents(dirs.root) == before


# --- A data directory is never replaced (T18) ------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["up", "tag"])
@pytest.mark.parametrize("there", ["seeding", "not made yet", "holding it"])
def test_an_album_named_like_a_trackers_root_never_replaces_it(monkeypatch, dirs, command, there) -> None:
    tracker_root = dirs.downloads / "red" / ("ops" if there == "holding it" else "")
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(tracker_root))
    if there != "not made yet":
        album(tracker_root / "Seeding")
    source = album(dirs.root / "in" / "Album")
    before = contents(dirs.downloads)
    asked = answer(monkeypatch, True, True)

    # salmon up gives the global directory as the parent of a first tracker without a key; salmon tag gives none.
    with pytest.raises(UploadError, match="which is a download_directory"):
        _rename(monkeypatch, source, "red", dirs.downloads if command == "up" else None)

    assert asked == []
    assert contents(dirs.downloads) == before


def test_salmon_tags_rename_is_otherwise_unchanged(monkeypatch, dirs, capsys) -> None:
    links_fail(monkeypatch)
    asked = answer(monkeypatch)

    made = _rename(monkeypatch, album(dirs.root / "in" / "Album"), "New", None)

    assert made == str(dirs.downloads / "New")
    assert "falling back to non-hardlink copy" in capsys.readouterr().out
    assert asked == []


# --- An album worked on as a copy (T19) ------------------------------------------------------------------------------


@pytest.mark.parametrize("why", ["library", "shares files"])
def test_a_copy_ends_in_the_trackers_directory_and_the_original_is_untouched(monkeypatch, dirs, why) -> None:
    if why == "library":
        monkeypatch.setattr(cfg.directory, "library_dirs", [str(dirs.library)])
        source = _album(dirs.library / "Album")
    else:
        source = _album(dirs.root / "in" / "Album")
        os.link(source / "cover.jpg", dirs.root / "seeded.jpg")
    before = contents(source), inodes(source)

    run, status = _up(monkeypatch, dirs, source, "RED")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert "The renamed copy stays where it is renamed to." in run.result.output
    assert (contents(source), inodes(source)) == before
    assert len(list((dirs.red / RENAMED).glob("*.flac"))) == 2
    assert not set(inodes(dirs.red / RENAMED).values()) & set(inodes(source).values())


def test_a_staged_copy_on_another_mount_asks_before_it_is_copied(monkeypatch, dirs) -> None:
    staged = album(dirs.downloads / ".salmon-staging" / "run-1" / "Album")
    links_fail(monkeypatch)
    asked = answer(monkeypatch, True)

    _rename(monkeypatch, staged, "New", dirs.red)

    assert "Copy instead" in asked[0][0]


# --- Later trackers (T20-T23) ----------------------------------------------------------------------------------------


@pytest.fixture
def seen(monkeypatch) -> SimpleNamespace:
    """What each torrent was made from, as it was then, and what each local client was given."""
    found = SimpleNamespace(torrents=[], saved=[])
    generate = upload_module.generate_torrent

    def generate_torrent(site: Any, path: str, *args: Any, **kwargs: Any):
        found.torrents.append((site.site_code, Path(path), contents(Path(path))))
        return generate(site, path, *args, **kwargs)

    def add_to_downloader(_client: Any, save_path: str, data: bytes, *, is_paused: bool, label: str) -> bool:
        torrent = Torrent.read_stream(data)
        found.saved.append((str(torrent.source), str(torrent.name), save_path))
        return True

    monkeypatch.setattr(upload_module, "generate_torrent", generate_torrent)
    monkeypatch.setattr(QBittorrentClient, "add_to_downloader", add_to_downloader)
    return found


def test_two_trackers_get_two_linked_folders_each_with_its_torrent_and_save_path(monkeypatch, dirs, seen) -> None:
    made: list[str] = []

    async def transcode(path: str, bitrate: str, *args: Any, output_dir: str | None = None, **kwargs: Any) -> str:
        made.append(str(output_dir))
        return await _transcode(path, bitrate, *args, output_dir=output_dir, **kwargs)

    source = _album(dirs.root / "in" / "Album")

    run, status = _up(
        monkeypatch,
        dirs,
        source,
        "RED",
        "OPS",
        conversions="V0",
        real_seedbox_queue=True,
        transcode_folder=transcode,
    )

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    converted = f"{RENAMED} [V0]"
    assert [(code, path) for code, path, _ in seen.torrents] == [
        ("RED", dirs.red / RENAMED),
        ("RED", dirs.red / converted),
        ("OPS", dirs.ops / RENAMED),
        ("OPS", dirs.ops / converted),
    ]
    assert inodes(dirs.ops / RENAMED) == inodes(dirs.red / RENAMED)
    assert inodes(dirs.ops / converted) == inodes(dirs.red / converted)
    assert made == [str(dirs.red)] * 2  # The second turn's converter finds the first's
    assert sorted(seen.saved) == sorted(
        [
            ("RED", RENAMED, str(dirs.red)),
            ("RED", converted, str(dirs.red)),
            ("OPS", RENAMED, str(dirs.ops)),
            ("OPS", converted, str(dirs.ops)),
        ]
    )
    for code, path, _files in seen.torrents:
        written = Torrent.read(dirs.torrents / f"{path.name} - {code}.torrent")
        assert written.verify(str(path))


def test_a_later_tracker_without_a_key_is_placed_in_the_global_directory(monkeypatch, dirs, seen) -> None:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED", "OPS", real_seedbox_queue=True)

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert seen.torrents[-1][:2] == ("OPS", dirs.downloads / RENAMED)
    assert ("OPS", RENAMED, str(dirs.downloads)) in seen.saved


def test_a_folder_is_placed_after_its_cover_and_never_written_into(monkeypatch, dirs, seen) -> None:
    given: list[Path] = []

    async def cover(site: Any, _group: Any, _urls: Any, path: str, *_args: Any, **_kwargs: Any) -> tuple[bool, None]:
        given.append(Path(path))
        # As a cover downloaded for this tracker's turn.
        (Path(path) / f"cover-{site.site_code}.jpg").write_bytes(b"downloaded")
        return True, None

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED", "OPS", resolve_cover_url=cover)

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert given == [dirs.red / RENAMED] * 2  # Always the first folder
    code, path, files = seen.torrents[-1]
    assert (code, path) == ("OPS", dirs.ops / RENAMED)
    assert "cover-OPS.jpg" in files  # Placed after its own cover step
    assert "Spectrals" not in files
    assert contents(path) == files


@pytest.mark.parametrize("refused", [False, True], ids=["first up", "first refused"])
def test_a_spectral_check_after_an_upload_writes_into_the_first_folder_only(monkeypatch, dirs, seen, refused) -> None:
    given: list[Path] = []

    async def check(_site: Any, path: str, *_args: Any, **_kwargs: Any) -> tuple[bool, None, None, None]:
        given.append(Path(path))
        # As an image the check made for this run.
        (Path(path) / "after upload.png").write_bytes(b"png")
        return False, None, None, None

    run, _status = _up(
        monkeypatch,
        dirs,
        _album(dirs.root / "in" / "Album"),
        "RED",
        "OPS",
        args=("--spectrals-after",),
        classes={"RED": _RefusingRed if refused else RedApi, "OPS": OpsApi},
        post_upload_spectral_check=check,
    )

    assert run.result.exit_code == 0, run.result.output
    assert given == [dirs.red / RENAMED]  # Also when OPS's is the first torrent up
    code, path, files = seen.torrents[-1]
    assert (code, path) == ("OPS", dirs.ops / RENAMED)
    assert contents(path) == files


class _RefusingRed(RedApi):
    async def upload(self, *_args: Any, **_kwargs: Any):
        raise RequestFailedError("refused")


def test_the_second_tracker_is_still_placed_when_the_first_upload_fails(monkeypatch, dirs, seen) -> None:
    run, status = _up(
        monkeypatch,
        dirs,
        _album(dirs.root / "in" / "Album"),
        "RED",
        "OPS",
        classes={"RED": _RefusingRed, "OPS": OpsApi},
    )

    assert status == 3, run.result.output
    assert seen.torrents[-1][:2] == ("OPS", dirs.ops / RENAMED)
    assert inodes(dirs.ops / RENAMED) == inodes(dirs.red / RENAMED)


def test_a_collision_under_yyy_skips_that_tracker_and_the_run_is_not_complete(monkeypatch, dirs, seen) -> None:
    before = contents(_earlier(dirs.ops / RENAMED))

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED", "OPS")

    assert status == 3, run.result.output  # RED is up
    assert "Not replacing" in run.result.output and "Skipping upload to OPS" in run.result.output
    assert "would you like to replace it?" not in run.result.output
    assert [code for code, _, _ in seen.torrents] == ["RED"]
    assert contents(dirs.ops / RENAMED) == before


def test_a_run_whose_only_possible_upload_is_skipped_exits_1(monkeypatch, dirs) -> None:
    _earlier(dirs.ops / RENAMED)

    run, status = _up(
        monkeypatch,
        dirs,
        _album(dirs.root / "in" / "Album"),
        "RED",
        "OPS",
        resolve_cover_url=_red_has_no_cover,
    )

    assert status == 1, run.result.output


async def _red_has_no_cover(site: Any, *_args: Any, **_kwargs: Any) -> tuple[bool, None]:
    return site.site_code != "RED", None


@pytest.mark.parametrize("error", [PlacementError("no space"), PlacementDeclined("not copying")])
def test_any_placement_error_skips_the_tracker_as_failed(monkeypatch, dirs, error) -> None:
    async def fail(*_args: Any) -> str:
        raise error

    monkeypatch.setattr(PlacedFolders, "place", fail)

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED")

    assert status == 1, run.result.output
    assert str(error) in run.result.output
    assert [sent for sent in run.tracker.not_gets() if sent.query.get("action") == "upload"] == []


@pytest.mark.parametrize("shared", [False, True], ids=["the source", "the original of a copy"])
def test_the_folder_the_run_started_from_is_never_replaced(monkeypatch, dirs, shared) -> None:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)
    source = _album(dirs.downloads / RENAMED)  # Where OPS's copy would go, under the same name
    if shared:
        os.link(source / "cover.jpg", dirs.root / "seeded.jpg")
    inode = (source / "cover.jpg").stat().st_ino

    async def yes(dest: str) -> Replacing:
        return Replacing.of(dest)  # As if the user agreed to replace it

    monkeypatch.setattr(placement, "_agree_to_replace", yes)

    run, status = _up(monkeypatch, dirs, source, "RED", "OPS")

    assert status == 3, run.result.output
    assert (source / "cover.jpg").stat().st_ino == inode
    assert "put OPS first with -t, or use salmon cross-upload" in run.result.output


def test_the_album_already_in_a_later_trackers_directory_is_used_without_asking(monkeypatch, dirs, seen) -> None:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.red))

    run, status = _up(monkeypatch, dirs, _album(dirs.root / "in" / "Album"), "RED", "OPS")

    assert (run.result.exit_code, status) == (0, 0), run.result.output
    assert [path for _, path, _ in seen.torrents] == [dirs.red / RENAMED] * 2


# --- Conversions (T24) -----------------------------------------------------------------------------------------------

CONVERSIONS = {
    "transcode": {"name": "MP3 V0", "action": "transcode", "encoding": "V0"},
    "downconversion": {
        "name": "16bit 44.1 kHz",
        "action": "downconvert",
        "target_bitdepth": 16,
        "target_sample_rate": 44100,
    },
}


@pytest.fixture
def converted(monkeypatch, dirs) -> SimpleNamespace:
    """The real converters writing files instead of encoding, and an upload that only queues its seedbox tasks."""
    found = SimpleNamespace(encoded=[], uploaded=[], saved=[], album=album(dirs.red / "Album [24bit FLAC]"))
    for track in found.album.glob("*.flac"):
        write_flac(track, bits=24, rate=44100, title=track.stem)

    async def encode(items: list[Any], _kind: Any) -> None:
        for item in items:
            found.encoded.append(item.dst)
            Path(item.dst).parent.mkdir(parents=True, exist_ok=True)
            if item.dst.endswith(".flac"):
                write_flac(Path(item.dst), bits=16, rate=44100)
            else:
                Path(item.dst).write_bytes(b"mp3")

    async def upload(site: Any, path: str, *args: Any, **_kwargs: Any) -> tuple[int, int, str, None, str]:
        found.uploaded.append((site.site_code, path))
        manager, torrent = args[-1], str(dirs.torrents / f"{site.site_code}.torrent")
        Path(torrent).write_bytes(b"torrent")
        manager.add_upload_task(torrent, "seed", False, folder=path, site_code=site.site_code)
        return 1, 5, torrent, None, "https://tracker.test/1"

    async def add(_client: Any, save_path: str, torrent: str, *_args: Any) -> bool:
        found.saved.append((Path(torrent).stem, save_path))
        return True

    monkeypatch.setattr(transcoding, "_transcode_audio_files", encode)
    monkeypatch.setattr(downconverting, "_convert_audio_files", encode)
    monkeypatch.setattr(conversion_upload, "upload_and_report", upload)
    monkeypatch.setattr(conversion_upload, "check_folder_structure", _returning_async())
    monkeypatch.setattr("salmon.uploader.seedbox._add_to_downloader", add)
    monkeypatch.setattr(QBittorrentClient, "login", lambda _client: None)
    monkeypatch.setattr(cfg, "seedbox", [Seedbox(name="box", torrent_client="qbittorrent+http://127.0.0.1:9")])
    found.placed = PlacedFolders(str(found.album))
    found.manager = UploadManager()
    return found


def _convert_for(found: SimpleNamespace, task: dict[str, Any], *codes: str) -> None:
    tracks = {track.name: {"precision": 24, "sample rate": 44100} for track in found.album.glob("*.flac")}
    for code in codes:
        site: Any = SimpleNamespace(site_code=code, site_string=code)
        anyio.run(
            partial(
                conversion_upload.execute_downconversion_tasks,
                [task],
                str(found.album),
                site,
                5,
                {"scene": False, "format": "FLAC", "encoding": "24bit Lossless"},
                None,
                tracks,
                False,
                False,
                None,
                None,
                None,
                None,
                None,
                found.manager,
                "WEB",
                "https://tracker.test/1",
                max_path_length=180,
                placed=found.placed,
            )
        )
    anyio.run(found.manager.execute_upload)


@pytest.mark.parametrize("kind", list(CONVERSIONS))
def test_a_conversion_is_encoded_once_placed_and_recorded_for_both_trackers(dirs, converted, kind) -> None:
    _convert_for(converted, CONVERSIONS[kind], "RED", "OPS")

    assert len(converted.encoded) == 2  # One encode of each track
    (_, red), (_, ops) = converted.uploaded
    assert (Path(red).parent, Path(ops).parent) == (dirs.red, dirs.ops)
    assert inodes(Path(ops)) == inodes(Path(red))
    record_there = conversion_of(ops)
    assert record_there == conversion_of(red)
    assert record_there is not None and record_there["source"] == str(converted.album)
    assert sorted(converted.saved) == [("OPS", str(dirs.ops)), ("RED", str(dirs.red))]


def test_a_conversion_is_made_for_the_first_tracker_that_uploads_it(dirs, converted) -> None:
    _convert_for(converted, CONVERSIONS["transcode"], "OPS", "DIC")

    assert [Path(path).parent for _, path in converted.uploaded] == [dirs.ops, dirs.downloads]
    assert all(Path(made).is_relative_to(dirs.ops) for made in converted.encoded)
    assert not [entry for entry in dirs.red.iterdir() if "MP3" in entry.name]  # No orphan for RED


def test_a_conversion_has_one_path_however_its_directory_is_written(monkeypatch, dirs, converted) -> None:
    # remove_local_after_copy goes by the path: two would let one tracker's remove what another's client seeds.
    for code in ("red", "ops"):
        monkeypatch.setattr(getattr(cfg.tracker, code), "download_directory", f"{dirs.red}//")
    client = "qbittorrent+http://127.0.0.1:9"
    boxes = [
        Seedbox(name="home", trackers=["RED"], torrent_client=client),
        Seedbox(
            name="nas", type="rclone", url="nas", trackers=["OPS"], remove_local_after_copy=True, torrent_client=client
        ),
    ]
    monkeypatch.setattr(cfg, "seedbox", boxes)
    monkeypatch.setattr("salmon.uploader.seedbox._rclone_upload_folder", _returning_async(True))
    converted.manager = UploadManager()

    _convert_for(converted, CONVERSIONS["transcode"], "RED", "OPS")
    (_, first), (_, second) = converted.uploaded
    converted.manager.add_upload_task(second, "folder", False, site_code="OPS")
    anyio.run(converted.manager.execute_upload)
    converted.manager.remove_local_copies()

    assert Path(first).is_dir()  # RED's local client seeds it
    assert first == second


def test_a_conversion_that_cannot_be_placed_is_skipped_and_the_others_go_on(monkeypatch, dirs, converted) -> None:
    _convert_for(converted, CONVERSIONS["transcode"], "RED")
    name = Path(converted.uploaded[0][1]).name
    before = contents(_earlier(dirs.ops / name))
    answer(monkeypatch, False)

    _convert_for(converted, CONVERSIONS["transcode"], "OPS", "DIC")

    assert [code for code, _ in converted.uploaded] == ["RED", "DIC"]
    assert contents(dirs.ops / name) == before


@pytest.mark.parametrize("holding", [False, True], ids=["a data directory", "holding one"])
def test_a_transcode_never_replaces_a_data_directory(monkeypatch, dirs, holding) -> None:
    source = album(dirs.root / "in" / "Album")
    root = dirs.downloads / "Album [MP3 V0]"
    tracker_root = root / "red" if holding else root
    monkeypatch.setattr(cfg.tracker.red, "download_directory", str(tracker_root))
    album(tracker_root / "Seeding")
    before = contents(dirs.downloads)
    asked = answer(monkeypatch, True)
    monkeypatch.setattr(transcoding, "_transcode_audio_files", _returning_async())

    with pytest.raises(UploadError, match="which is a download_directory"):
        anyio.run(partial(transcoding.transcode_folder, str(source), "V0", output_dir=str(dirs.downloads)))

    assert asked == []
    assert contents(dirs.downloads) == before


# --- Dry runs and requests (T28, R19) --------------------------------------------------------------------------------


@pytest.mark.parametrize("exists", [False, True], ids=["nothing there", "folders there"])
def test_a_dry_run_makes_no_directory_leaves_no_file_and_never_asks_to_replace(monkeypatch, dirs, exists) -> None:
    source = _album(dirs.root / "in" / "Album")
    if exists:
        for name in (RENAMED, f"{RENAMED} [V0]"):
            _earlier(dirs.red / name)
            _earlier(dirs.ops / name)
    (dirs.downloads / ".salmon-staging").mkdir()  # Where a dry run's own directory comes and goes
    before = contents(dirs.root)

    run, _status = _up(monkeypatch, dirs, source, "RED", "OPS", conversions="V0", args=("--dry-run",))

    assert run.result.exit_code == 0, run.result.output
    assert contents(dirs.root) == before
    assert "would you like to replace it?" not in run.result.output
    # Once for each tracker's album: a conversion that is there is found again, not asked about.
    assert run.result.output.count("a real run would ask before replacing it") == (2 if exists else 0)
    assert "Created" not in run.result.output


def test_a_key_changes_no_request(monkeypatch, dirs, images) -> None:
    def requests(source: Path) -> list[tuple[Any, ...]]:
        run, status = _up(monkeypatch, dirs, source, "RED", "OPS", conversions="V0")
        assert (run.result.exit_code, status) == (0, 0), run.result.output
        return [
            (
                sent.method,
                sent.path,
                sent.query,
                [(name, _comparable(name, value, run.tracker.url)) for name, value in sent.fields],
            )
            for sent in run.tracker.sent
        ]

    with_keys = requests(_album(dirs.root / "keys" / "Album"))
    images.clear()  # Its image URLs are numbered
    for folder in (dirs.red, dirs.ops):
        shutil.rmtree(folder)
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)

    assert requests(_album(dirs.root / "no keys" / "Album")) == with_keys


def _comparable(name: str, value: Any, site: str) -> Any:
    """A field as two runs can share it: each fake has a port of its own, and a torrent's bytes its creation date."""
    if name == "file_input":
        return [(str(file), file.size) for file in Torrent.read_stream(value[1]).files]
    return value.replace(site, "TRACKER") if isinstance(value, str) else value


# --- The other deletes ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("conversions", ["none", "V0"])
def test_remove_source_dir_keeps_the_album_a_tracker_was_skipped_over(monkeypatch, dirs, conversions) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    source = _album(dirs.ops / RENAMED)  # Where OPS's copy would go: never replaced (D2)

    run, status = _up(monkeypatch, dirs, source, "RED", "OPS", conversions=conversions)

    assert status == 3, run.result.output
    assert len(list(source.glob("*.flac"))) == 2
    assert f"remove_source_dir is on, but {source} is kept" in run.result.output


def test_remove_source_dir_still_removes_a_source_no_tracker_was_skipped_over(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    source = _album(dirs.root / "in" / "Album")
    _earlier(dirs.ops / RENAMED)  # OPS is skipped, over another folder

    run, status = _up(monkeypatch, dirs, source, "RED", "OPS")

    assert status == 3, run.result.output
    assert not source.exists()


def test_remove_source_dir_keeps_an_album_the_skipped_trackers_folder_links_to(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    source = _album(dirs.root / "in" / "Album")
    dirs.ops.mkdir(parents=True)
    (dirs.ops / RENAMED).symlink_to(source, target_is_directory=True)

    run, status = _up(monkeypatch, dirs, source, "RED", "OPS")

    assert status == 3, run.result.output
    assert len(list((dirs.ops / RENAMED).glob("*.flac"))) == 2  # The link still leads to the album
    assert f"remove_source_dir is on, but {source} is kept" in run.result.output


@pytest.mark.parametrize("holding", [False, True], ids=["a data directory", "holding one"])
def test_remove_source_dir_never_removes_a_data_directory(monkeypatch, dirs, holding) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    given = dirs.ops
    if holding:
        monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.ops / "ops"))
    _album(given)
    final = album(dirs.red / RENAMED)
    before = contents(given)

    anyio.run(partial(remove_source_folder, str(given), str(final), only_here=[]))

    assert contents(given) == before


@pytest.mark.parametrize("holding", [False, True], ids=["a data directory", "holding one"])
def test_delete_music_folder_never_deletes_a_data_directory(monkeypatch, dirs, holding) -> None:
    given = album(dirs.ops)
    if holding:
        monkeypatch.setattr(cfg.tracker.ops, "download_directory", str(dirs.ops / "ops"))
    before = contents(given)
    asked = answer(monkeypatch, True)

    anyio.run(partial(salmon.uploader._delete_after_abort, str(given), str(given), None, None, ask_again=True))

    assert asked == []
    assert contents(given) == before


def test_with_no_key_both_deletes_are_as_before(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)
    source, deleted = album(dirs.root / "source"), album(dirs.root / "deleted")

    anyio.run(partial(remove_source_folder, str(source), str(album(dirs.downloads / "New")), only_here=[]))
    anyio.run(partial(salmon.uploader._delete_after_abort, str(deleted), str(deleted), None, None))

    assert not source.exists() and not deleted.exists()


def test_a_rename_clears_a_stale_record_at_its_name(monkeypatch, dirs) -> None:
    dirs.red.mkdir(parents=True)
    record_conversion(str(dirs.red / "New"), source="/gone", kind="transcode", bitrate="V0")

    _rename(monkeypatch, album(dirs.root / "in" / "New"), "New", dirs.red)

    assert conversion_of(str(dirs.red / "New")) is None


def test_a_rename_whose_record_cannot_be_written_is_undone(monkeypatch, dirs) -> None:
    source = album(dirs.root / "in" / "New")
    record_conversion(str(source), source="/the/album", kind="transcode", bitrate="V0")
    before = contents(_earlier(dirs.red / "New"))
    records = dirs.red / ".salmon-conversions"
    records.mkdir()
    records.chmod(0o555)
    answer(monkeypatch, True)
    try:
        with pytest.raises(UploadError, match="Could not make"):
            _rename(monkeypatch, source, "New", dirs.red)
    finally:
        records.chmod(0o755)

    assert contents(dirs.red / "New") == before


def test_a_conversion_whose_output_is_refused_leaves_the_upload_incomplete(monkeypatch, dirs) -> None:
    async def claim(folder: str) -> str | None:
        return "another job works on it" if "MP3" in folder else None

    with placement.claiming(claim):
        run, status = _up(
            monkeypatch,
            dirs,
            _album(dirs.root / "in" / "Album"),
            "RED",
            conversions="V0",
            transcode_folder=transcoding.transcode_folder,
        )

    assert status == 3, run.result.output
    assert "another job works on it" in run.result.output
    assert not [entry for entry in dirs.red.iterdir() if "MP3" in entry.name]
