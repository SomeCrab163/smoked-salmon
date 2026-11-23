"""Per-tracker download directories for tests (#1056): configured folders, and what a run left in them."""

import errno
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from salmon import cfg, interaction
from salmon.placement import BUILDING_PREFIX


def tracker_dirs(monkeypatch, tmp_path: Path, *codes: str) -> SimpleNamespace:
    """Configure a download_directory, and one for each of `codes` (RED and OPS by default) that is not made yet."""
    dirs = SimpleNamespace(
        root=tmp_path,
        downloads=tmp_path / "downloads",
        red=tmp_path / "data" / "red",
        ops=tmp_path / "data" / "ops",
        dic=tmp_path / "data" / "dic",
        library=tmp_path / "library",
    )
    dirs.downloads.mkdir(exist_ok=True)
    dirs.library.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg.directory, "download_directory", str(dirs.downloads))
    monkeypatch.setattr(cfg.directory, "library_dirs", [])
    monkeypatch.setattr(cfg.directory, "tmp_dir", None)
    monkeypatch.setattr(cfg.directory, "hardlinks", True)
    monkeypatch.setattr(cfg.upload, "yes_all", False)
    chosen = [code.lower() for code in (codes or ("RED", "OPS"))]
    for code in ("red", "ops", "dic"):
        folder = str(getattr(dirs, code)) if code in chosen else None
        monkeypatch.setattr(getattr(cfg.tracker, code), "download_directory", folder)
    return dirs


def answer(monkeypatch, *answers: bool) -> list[tuple[str, dict[str, Any]]]:
    """Answer the confirms in order; each question asked, with its arguments. One question too many fails.

    A rename's question whether to take the new name is always agreed to, and is neither of them.
    """
    left, asked = list(answers), []

    async def confirm(text: str, **kwargs: Any) -> bool:
        if "replace the original folder name?" in text:
            return True
        asked.append((text, kwargs))
        assert left, f"a question too many: {text}"
        return left.pop(0)

    monkeypatch.setattr(interaction, "confirm", confirm)
    return asked


def links_fail(monkeypatch, code: int = errno.EXDEV, *, after: int = 0, winerror: int | None = None) -> None:
    """os.link fails with `code` once it has made `after` links, as across two bind mounts of one filesystem."""
    real, calls = os.link, []

    def link(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        calls.append(src)
        if len(calls) > after:
            error = OSError(code, os.strerror(code), src)
            if winerror is not None:
                error.winerror = winerror
            raise error
        real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)


def contents(folder: Path) -> dict[str, str]:
    """A hash of each file under folder, "dir", or where a link points, by relative path."""

    def what(entry: Path) -> str:
        if entry.is_symlink():
            return f"-> {os.readlink(entry)}"
        return "dir" if entry.is_dir() else hashlib.sha256(entry.read_bytes()).hexdigest()

    return {entry.relative_to(folder).as_posix(): what(entry) for entry in sorted(folder.rglob("*"))}


def inodes(folder: Path) -> dict[str, int]:
    """The inode of each file under folder, by relative path."""
    return {entry.relative_to(folder).as_posix(): entry.stat().st_ino for entry in folder.rglob("*") if entry.is_file()}


def building_left(root: Path) -> list[Path]:
    """The folders placement left half built anywhere under root."""
    return list(root.rglob(f"{BUILDING_PREFIX}*"))
