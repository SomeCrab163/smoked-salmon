"""Re-running the metadata step never deletes the album the run started from (found with #1056)."""

import copy
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from helpers.album_files import album
from helpers.uploader_dry_run import METADATA, _returning, _returning_async
from helpers.uploader_fakes import stub_uploader

import salmon.tagger.foldername
from salmon import cfg, interaction
from salmon.errors import UploadError
from salmon.uploader import metadata_review, staging


@pytest.fixture
def world(monkeypatch, tmp_path) -> SimpleNamespace:
    """The folder albums are renamed into, also reachable as "linked", and a metadata step that is run again."""
    found = SimpleNamespace(root=tmp_path, real=tmp_path / "real", linked=tmp_path / "linked")
    found.real.mkdir()
    found.linked.symlink_to(found.real, target_is_directory=True)
    (tmp_path / "global").mkdir()
    monkeypatch.setattr(cfg.directory, "download_directory", str(tmp_path / "global"))
    monkeypatch.setattr(cfg.directory, "library_dirs", [])
    monkeypatch.setattr(cfg.directory, "tmp_dir", None)
    monkeypatch.setattr(cfg.upload, "yes_all", False)
    metadata = copy.deepcopy(METADATA)
    stub_uploader(
        monkeypatch,
        {
            "review_metadata_with_ai": _returning_async(metadata),
            "tag_files": _returning_async(),
            "check_tags": _returning_async({}),
            "rename_files": _returning_async(),
            "check_folder_structure": _returning_async(),
            "gather_audio_info": _returning({}),
            "gather_tags": _returning({}),
        },
    )
    monkeypatch.setattr(metadata_review, "process_tag_issues", _returning([]))
    return found


def _review(monkeypatch, given: str, names: list[str], parent: str | None, original: str | None = None) -> None:
    """Review, re-running the metadata step until each of `names` was the folder's, agreeing to all else."""
    left = iter(names)
    monkeypatch.setattr(salmon.tagger.foldername, "generate_folder_name", lambda _metadata: next(left))
    passes: list[str] = []

    async def confirm(text: str, **_kwargs: Any) -> bool:
        if "upload the torrent" in text:
            passes.append(text)
            return len(passes) == len(names)
        return True

    monkeypatch.setattr(interaction, "confirm", confirm)
    anyio.run(
        partial(
            metadata_review.edit_metadata,
            given,
            {},
            copy.deepcopy(METADATA),
            None,
            "WEB",
            {},
            recompress=False,
            auto_rename=True,
            spectral_ids=None,
            skip_integrity_check=True,
            rename_into=parent,
            original=original,
        )
    )


@pytest.mark.parametrize("spelled", ["album", "directory"], ids=["album through a link", "directory through a link"])
def test_the_source_spelled_another_way_is_not_removed_by_a_rerun(monkeypatch, world, spelled) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    source = album(world.real / "Album")
    given, parent = (world.linked, world.real) if spelled == "album" else (world.real, world.linked)

    # The first pass renames the album into itself, under the directory's spelling.
    _review(monkeypatch, str(given / "Album"), ["Album", "New"], str(parent))

    assert sorted(file.name for file in source.iterdir()) == ["01.flac", "02.flac"]
    assert (world.real / "New" / "01.flac").is_file()


def test_the_folder_an_earlier_pass_made_is_still_removed(monkeypatch, world) -> None:
    monkeypatch.setattr(cfg.upload.formatting, "remove_source_dir", True)
    source = album(world.real / "Album")

    _review(monkeypatch, str(source), ["First", "Second"], str(world.real))

    assert sorted(entry.name for entry in world.real.iterdir() if entry.name[0] != ".") == ["Album", "Second"]


@pytest.mark.parametrize("names", [["New", "Album"], ["New", "Newer", "Album"]], ids=["second pass", "third pass"])
def test_the_album_a_copy_is_of_is_never_replaced_on_a_later_pass(monkeypatch, world, names) -> None:
    original = album(world.real / "Album")
    seeded = Path(world.root / "seeded.flac")
    os.link(original / "01.flac", seeded)  # So the run works on a copy

    with (
        pytest.raises(UploadError, match="it is the album this run works on a copy of"),
        staging.staged_source(str(original), scratch=False) as (staged, _),
    ):
        _review(monkeypatch, staged, names, str(world.real), str(original))

    assert (original / "01.flac").samefile(seeded)
