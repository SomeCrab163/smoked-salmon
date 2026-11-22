"""salmon cross-upload with per-tracker directories (#1056): where the source is found, and where TARGET seeds from."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from helpers.album_files import album
from helpers.cross_upload import (
    FOLDER,
    TORRENT_ID,
    _album,
    _converted,
    _cross_upload,
    _ops_answer,
    _snapshot,
    _transcode,
)
from helpers.tracker_dirs import contents, inodes, links_fail, tracker_dirs
from helpers.uploader_fakes import stub_uploader
from test_cross_upload import (
    _source_has,
    settings,  # noqa: F401 (a fixture)
)
from torf import Torrent

import salmon.cross_upload as cross_upload_module
from salmon import cfg, interaction
from salmon.config.validations import Seedbox
from salmon.cross_upload import CrossUploadRefused, _release_path
from salmon.uploader.seedbox import UploadManager

pytestmark = pytest.mark.usefixtures("settings")


@pytest.fixture
def dirs(monkeypatch, tmp_path) -> SimpleNamespace:
    """OPS, the source, and RED, the target, each with a directory of its own. RED's is not made yet."""
    found = tracker_dirs(monkeypatch, tmp_path)
    found.torrents = tmp_path / "torrents"
    found.torrents.mkdir()
    return found


def _local_seedbox(monkeypatch, *albums: Path):
    """prepare: SOURCE holds the albums, and the seedbox is a local one without a directory."""
    has = _source_has(*albums)

    def prepare(source: Any, target: Any) -> None:
        has(source, target)
        monkeypatch.setattr(cfg, "seedbox", [Seedbox(name="box", torrent_client="qbittorrent+http://127.0.0.1:9")])

    return prepare


@pytest.fixture
def made_from(monkeypatch) -> list[Path]:
    """The folder each FLAC torrent was made from."""
    found: list[Path] = []
    generate = cross_upload_module.generate_torrent

    def generate_torrent(site: Any, path: str, **kwargs: Any):
        found.append(Path(path))
        return generate(site, path, **kwargs)

    monkeypatch.setattr(cross_upload_module, "generate_torrent", generate_torrent)
    return found


# --- Finding the source (T25) ----------------------------------------------------------------------------------------


def _found(path: Path | None = None) -> Path:
    return _release_path({"torrent": {"filePath": "Album"}}, str(path) if path else None, "OPS")


def test_path_wins_over_both_directories(dirs) -> None:
    album(dirs.ops / "Album")
    album(dirs.downloads / "Album")
    given = album(dirs.root / "given" / "Album")

    assert _found(given) == given.resolve()


def test_the_sources_directory_comes_before_the_global_one(dirs) -> None:
    there = album(dirs.ops / "Album")
    album(dirs.downloads / "Album")

    assert _found() == there.resolve()


def test_the_global_directory_is_next(dirs) -> None:
    album(dirs.red / "Album")  # TARGET's is never looked in
    there = album(dirs.downloads / "Album")

    assert _found() == there.resolve()


def test_not_found_lists_each_place_tried(dirs) -> None:
    with pytest.raises(CrossUploadRefused) as refused:
        _found()

    expected = f"the files are not at {dirs.ops / 'Album'} or at {dirs.downloads / 'Album'}"
    assert str(refused.value) == f"{expected}: give the album folder with --path"


def test_one_directory_is_tried_once(monkeypatch, dirs) -> None:
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)

    with pytest.raises(CrossUploadRefused) as refused:
        _found()

    assert str(refused.value) == f"the files are not at {dirs.downloads / 'Album'}: give the album folder with --path"


@pytest.mark.parametrize(
    ("root", "setting"),
    [("ops", "tracker.ops.download_directory"), ("downloads", "download_directory")],
)
def test_a_link_out_of_either_directory_is_refused(dirs, root, setting) -> None:
    outside = album(dirs.root / "outside" / "Album")
    getattr(dirs, root).mkdir(parents=True, exist_ok=True)
    (getattr(dirs, root) / "Album").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CrossUploadRefused, match=f"outside {setting}$"):
        _found()


def test_a_folder_that_fails_its_check_is_refused_without_trying_the_next(monkeypatch, dirs) -> None:
    good = _album(dirs.downloads / FOLDER)
    wrong = _album(dirs.ops / FOLDER)
    (wrong / "extra.txt").write_text("not in the torrent")

    run = _cross_upload(monkeypatch, dirs, [str(TORRENT_ID), "-yyy"], input="\n", prepare=_source_has(good))

    assert run.result.exit_code == 1, run.output
    assert f"the files at {wrong} are not the torrent's" in run.output
    assert run.target.posts() == []


# --- Into TARGET's directory (T26) -----------------------------------------------------------------------------------


@pytest.mark.parametrize("linked", [True, False], ids=["link", "copy"])
def test_the_release_goes_into_the_targets_directory_and_the_source_is_untouched(
    monkeypatch, dirs, made_from, linked
) -> None:
    monkeypatch.setattr(cfg.directory, "hardlinks", linked)
    source = _album(dirs.ops / FOLDER)
    before = _snapshot(source)
    converted_into: list[tuple[str, str | None]] = []
    queued: list[str | None] = []
    add_upload_task = UploadManager.add_upload_task

    def queue(manager: UploadManager, *args: Any, folder: str | None = None, **kwargs: Any) -> None:
        queued.append(folder)
        add_upload_task(manager, *args, folder=folder, **kwargs)

    monkeypatch.setattr(UploadManager, "add_upload_task", queue)

    async def transcode(path: str, bitrate: str, *args: Any, output_dir: str | None = None, **kwargs: Any) -> str:
        converted_into.append((path, output_dir))
        return await _transcode(path, bitrate, *args, output_dir=output_dir, **kwargs)

    run = _cross_upload(
        monkeypatch,
        dirs,
        [str(TORRENT_ID), "--transcode", "V0", "-yyy"],
        input="\n",
        prepare=_local_seedbox(monkeypatch, source),
        transcode=transcode,
    )

    assert run.result.exit_code == 0, run.output
    dest = dirs.red / FOLDER
    assert f"to: {dest}" in run.output
    assert made_from == [dest]
    assert queued[0] == str(dest)
    assert contents(dest) == contents(source)
    assert (inodes(dest) == inodes(source)) is linked
    assert _snapshot(source) == before
    assert converted_into == [(str(source), str(dirs.red))]  # From the source, into TARGET's
    assert sorted(run.seeded) == sorted([(str(dirs.red), FOLDER), (str(dirs.red), f"{FOLDER} [V0]")])
    assert "check that the seedbox's torrent client seeds from there" not in run.output
    assert run.copied == []  # As before: no folder copy to a seedbox
    assert Torrent.read(dirs.torrents / f"{FOLDER} - RED.torrent").verify(str(dest))


def test_a_downconversion_goes_into_the_targets_directory(monkeypatch, dirs) -> None:
    source = _album(dirs.ops / FOLDER, 96000, bits=24)

    def answer_24bit(folder: Path, torrent_id: int = TORRENT_ID) -> dict[str, Any]:
        response = _ops_answer(folder, torrent_id)
        response["torrent"]["encoding"] = "24bit Lossless"
        return response

    stub_uploader(monkeypatch, {"convert_folder": _converted})
    run = _cross_upload(
        monkeypatch,
        dirs,
        [str(TORRENT_ID), "--downconvert", "-yyy"],
        input="\n",
        prepare=_source_has(source, answer=answer_24bit),
    )

    assert run.result.exit_code == 0, run.output
    assert (dirs.red / f"{FOLDER} [16-48000]").is_dir()


def test_rclone_is_unchanged(monkeypatch, dirs) -> None:
    run = _cross_upload(
        monkeypatch, dirs, [str(TORRENT_ID), "-yyy"], input="\n", prepare=_source_has(_album(dirs.ops / FOLDER))
    )

    assert run.result.exit_code == 0, run.output
    assert run.seeded == [("/seed", FOLDER)]
    assert run.copied == []


def test_a_release_already_in_the_targets_directory_is_used_as_it_is(monkeypatch, dirs, made_from) -> None:
    source = _album(dirs.red / FOLDER)

    run = _cross_upload(
        monkeypatch, dirs, [str(TORRENT_ID), "--path", str(source), "-yyy"], input="\n", prepare=_source_has(source)
    )

    assert run.result.exit_code == 0, run.output
    assert made_from == [source]
    assert "   to: " not in run.output


def test_a_key_changes_no_request(monkeypatch, dirs) -> None:
    source = _album(dirs.downloads / FOLDER)
    args = [str(TORRENT_ID), "-yyy"]
    with_keys = _cross_upload(monkeypatch, dirs, args, input="\n", prepare=_source_has(source))
    monkeypatch.setattr(cfg.tracker.red, "download_directory", None)
    monkeypatch.setattr(cfg.tracker.ops, "download_directory", None)

    without = _cross_upload(monkeypatch, dirs, args, input="\n", prepare=_source_has(source))

    assert (with_keys.result.exit_code, without.result.exit_code) == (0, 0)
    assert [sent.step for sent in with_keys.source.sent] == [sent.step for sent in without.source.sent]
    assert [sent.step for sent in with_keys.target.sent] == [sent.step for sent in without.target.sent]
    sent_with, sent_without = with_keys.target.posts(), without.target.posts()
    for one, other in zip(sent_with, sent_without, strict=True):
        for name in set(one.fields) | set(other.fields):
            if name != "release_desc":  # Holds its run's fake SOURCE's port
                assert one.fields[name] == other.fields[name], name


# --- Declined (T27) --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("why", ["collision", "copy declined"])
def test_a_declined_release_is_skipped_the_next_goes_and_the_run_fails(monkeypatch, dirs, why) -> None:
    first = _album(dirs.ops / FOLDER)
    second = _album(dirs.ops / f"{FOLDER} 2")
    args = [str(TORRENT_ID), str(TORRENT_ID + 1)]
    if why == "collision":
        album(dirs.red / FOLDER)
        args.append("-yyy")
    else:
        links_fail(monkeypatch)
        copies = [False, True]  # Don't copy the first, copy the second

        async def confirm(text: str, **kwargs: Any) -> bool:
            return copies.pop(0) if "Copy instead" in text else bool(kwargs.get("default"))

        monkeypatch.setattr(interaction, "confirm", confirm)
    kept = contents(dirs.red) if dirs.red.exists() else {}

    run = _cross_upload(monkeypatch, dirs, args, input="\n\n\n\n", prepare=_source_has(first, second))

    assert run.result.exit_code == 1, run.output
    assert f"Skipping {TORRENT_ID}:" in run.output
    [post] = run.target.posts()
    assert Torrent.read_stream(post.files["file_input"][0]).name == second.name
    assert run.seeded == [("/seed", second.name)]  # The queue ran before the run failed
    assert {name: what for name, what in contents(dirs.red).items() if not name.startswith(second.name)} == kept


def test_a_placement_error_stops_the_run(monkeypatch, dirs) -> None:
    first = _album(dirs.ops / FOLDER)
    second = _album(dirs.ops / f"{FOLDER} 2")
    links_fail(monkeypatch, 13)

    run = _cross_upload(
        monkeypatch,
        dirs,
        [str(TORRENT_ID), str(TORRENT_ID + 1), "-yyy"],
        input="\n\n",
        prepare=_source_has(first, second),
    )

    assert run.result.exit_code == 1, run.output
    assert "Stopping: Could not make" in run.output
    assert run.target.posts() == []


def test_a_run_never_replaces_the_source_of_a_release_still_to_go(monkeypatch, dirs) -> None:
    # Two SOURCE torrents of one folder name: the second's files are where the first's copy would go.
    first = _album(dirs.ops / FOLDER)
    kept = _album(dirs.red / FOLDER)
    inode = os.stat(kept / "cover.jpg").st_ino
    found = iter([first, kept])

    async def yes(dest: str):
        return cross_upload_module.placement.Replacing.of(dest)

    monkeypatch.setattr(cross_upload_module.placement, "_agree_to_replace", yes)
    monkeypatch.setattr(cross_upload_module, "_release_path", lambda *_args: next(found))

    run = _cross_upload(
        monkeypatch,
        dirs,
        [str(TORRENT_ID), str(TORRENT_ID + 1), "-yyy"],
        input="\n\n",
        prepare=_source_has(first, kept),
    )

    assert run.result.exit_code == 1, run.output
    assert os.stat(kept / "cover.jpg").st_ino == inode


def test_the_destination_is_claimed_before_it_may_be_replaced(monkeypatch, dirs) -> None:
    source = _album(dirs.ops / FOLDER)
    before = contents(album(dirs.red / FOLDER))
    claimed: list[Path] = []
    asked: list[str] = []

    async def claim(folder: Path) -> str | None:
        claimed.append(folder)
        return "another job works on it" if folder == dirs.red / FOLDER else None

    async def confirm(text: str, **kwargs: Any) -> bool:
        asked.append(text)
        return bool(kwargs.get("default"))

    run_cross_upload = cross_upload_module.run_cross_upload
    monkeypatch.setattr(
        cross_upload_module, "run_cross_upload", lambda options: run_cross_upload(options, claim_folder=claim)
    )
    monkeypatch.setattr(interaction, "confirm", confirm)

    run = _cross_upload(monkeypatch, dirs, [str(TORRENT_ID)], input="\n", prepare=_source_has(source))

    assert run.result.exit_code == 1, run.output
    assert claimed == [source.resolve(), dirs.red / FOLDER]  # salmon web's lock takes both
    assert not [text for text in asked if "would you like to replace it?" in text]
    assert contents(dirs.red / FOLDER) == before
    assert run.target.posts() == []


# --- Dry runs (T28) --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("exists", [False, True], ids=["nothing there", "a folder there"])
def test_a_dry_run_makes_nothing_and_never_asks_to_replace(monkeypatch, dirs, exists) -> None:
    source = _album(dirs.ops / FOLDER)
    if exists:
        album(dirs.red / FOLDER)
    (dirs.downloads / ".salmon-staging").mkdir()  # Where a dry run's own directory comes and goes
    before = contents(dirs.root)

    run = _cross_upload(
        monkeypatch,
        dirs,
        [str(TORRENT_ID), "--transcode", "V0", "--dry-run", "-yyy"],
        input="\n",
        prepare=_source_has(source),
        transcode=_transcode,
    )

    assert run.result.exit_code == 0, run.output
    assert contents(dirs.root) == before
    assert f"to: {dirs.red / FOLDER}" in run.output
    assert ("already exists" in run.output) is exists
    assert "would you like to replace it?" not in run.output
    assert run.target.posts() == []


def test_the_plan_is_shown_for_a_targets_directory_salmon_may_not_read(monkeypatch, dirs) -> None:
    source = _album(dirs.ops / FOLDER)
    dirs.red.mkdir()
    dirs.red.chmod(0o000)
    try:
        run = _cross_upload(
            monkeypatch, dirs, [str(TORRENT_ID), "--dry-run", "-yyy"], input="\n", prepare=_source_has(source)
        )
    finally:
        dirs.red.chmod(0o755)

    assert run.result.exit_code == 0, run.output
    assert f"to: {dirs.red / FOLDER}" in run.output
