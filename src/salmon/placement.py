"""Putting an album folder into a tracker's download_directory: hardlinked or copied, whole or not at all (#1056).

The tagger and the uploader both use it, so it imports neither.

A folder is built in a hidden folder beside its destination and renamed into place. One the user agreed to replace is
only moved aside for that, to a hidden name beside it: it is put back if the placement fails, and removed only once
the new one is in place. Whether a link can be made is os.link's answer: st_dev is only a fast check before it, as two
bind mounts of one filesystem have the same st_dev and take no link between them.
"""

import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Self

import asyncclick as click

from salmon import cfg, dryrun, interaction
from salmon.common.files import shares_files
from salmon.config.validations import holds
from salmon.errors import UploadError
from salmon.hardlinks import cannot_link

# Hidden, so neither salmon web's picker nor a torrent client's watcher takes a folder being built for an album.
BUILDING_PREFIX = ".salmon-placing-"
REPLACED_PREFIX = ".salmon-replaced-"
SHARED_NOTE = " (it shares files with another folder through a hardlink or symlink: a torrent may seed from it)"

Ignore = Callable[[str, list[str]], Iterable[str]]
# Takes a destination for the running job before it may be replaced (salmon web's lock): why it can't, or None.
Claim = Callable[[str], Awaitable[str | None]]
_claimer: ContextVar[Claim | None] = ContextVar("placement_claim", default=None)


class PlacementError(UploadError):
    """The folder could not be made."""


class PlacementDeclined(PlacementError):
    """The folder was not made: the user declined, or it must never be replaced.

    Attributes:
        kept: It is refused for a folder this run works from, such as the album it started from.
    """

    def __init__(self, message: str, *, kept: bool = False) -> None:
        super().__init__(message)
        self.kept = kept


class _LinksUnavailable(Exception):
    """Not an OSError: copytree would collect one and go on (and lose its errno)."""


@contextmanager
def claiming(take: Claim) -> Iterator[None]:
    """Run the block, and every task it starts, with each destination taken through `take` before it may be replaced.

    The command, alone at its terminal, sets none and needs no lock.
    """
    token = _claimer.set(take)
    try:
        yield
    finally:
        _claimer.reset(token)


async def claim(dest: str) -> None:
    """Take dest for the running job.

    Raises:
        PlacementDeclined: Another job works on it, in it or on a folder holding it.
    """
    if (take := _claimer.get()) is not None and (refusal := await take(dest)) is not None:
        raise PlacementDeclined(f"Not using {dest}: {refusal}.")


def ensure_directory(folder: str) -> None:
    """mkdir -p, saying so when it creates. Nothing in a dry run."""
    if dryrun.active() or os.path.isdir(folder):
        return
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as error:
        raise PlacementError(f"Could not create {folder}: {error}") from error
    click.secho(f"Created {folder}.", fg="cyan")


@dataclass(frozen=True)
class Replacing:
    """The folder the user is asked whether to replace."""

    folder: os.stat_result  # As it was before the question
    shared: bool  # shares_files(dest) then

    @classmethod
    def of(cls, dest: str) -> Self:
        return cls(os.lstat(dest), shares_files(dest))

    def changed(self, dest: str) -> str | None:
        """Why the folder at dest is no longer the one asked about, or None if it is."""
        try:
            now = os.lstat(dest)
        except FileNotFoundError:
            return "it went away"
        # The times too: ext4 gives the inode of a deleted folder straight to the next one made. They also
        # change when an entry is added to or removed from the folder.
        if not os.path.samestat(now, self.folder) or (now.st_mtime_ns, now.st_ctime_ns) != (
            self.folder.st_mtime_ns,
            self.folder.st_ctime_ns,
        ):
            return "it changed"
        if not self.shared and shares_files(dest):
            return "it started sharing files with another folder"
        return None


def data_directory_refusal(path: str) -> str | None:
    """Why path must never be replaced, when it is or holds a data directory of a config with per-tracker ones."""
    if cfg.per_tracker_directories() and (folder := cfg.data_directory_inside(path)) is not None:
        return f"it is or holds {folder}, which is a download_directory"
    return None


def never_replaced(dest: str) -> str | None:
    """Why no folder may be made or replaced at dest, or None. Checked whether or not it exists yet."""
    # Also for a path not there: a symlinked parent can lead into a library.
    if cfg.directory.protects(dest):
        return "it is in library_dirs, or holds one"
    if (reason := data_directory_refusal(dest)) is not None:
        return reason
    if os.path.islink(dest) or os.path.isjunction(dest):
        return "it is a link"
    if os.path.lexists(dest) and not os.path.isdir(dest):
        return "it is not a folder"
    return None


def overlap(dest: str, *folders: str) -> str | None:
    """Why dest must not be made or replaced for being, holding or being inside one of folders, or None."""
    for folder in folders:
        if holds(dest, folder):
            return f"it is or holds {folder}, which this run works from"
        if holds(folder, dest):
            return f"it is inside {folder}, which this run works from"
    return None


def destination(src: str, code: str) -> str | None:
    """Where tracker `code`'s copy of the folder src goes, or None when the tracker uses src itself."""
    if not cfg.per_tracker_directories():
        return None
    dest = os.path.join(cfg.download_directory_for(code), os.path.basename(os.path.normpath(src)))
    return None if _same_folder(src, dest) else dest


def _same_folder(first: str, second: str) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


async def place_folder(
    src: str, dest: str, *, ignore: Ignore | None = None, replacing: Replacing | None = None
) -> None:
    """Make dest from src whole or not at all, its conversion record included: src's, or none. `replacing` is the
    folder the user agreed to replace.

    Raises:
        PlacementDeclined: dest must not be made or replaced, it is not as it was, or a copy was declined.
        PlacementError: It could not be made.
    """
    dryrun.refuse(f"put {src} at {dest}")
    _refuse(src, dest)
    await claim(dest)
    parent = os.path.dirname(dest) or os.curdir
    ensure_directory(parent)
    try:
        with _building_in(parent) as temp:
            built = os.path.join(temp, os.path.basename(dest))
            linked = await _build(src, built, dest, ignore)
            # Again: a job can have started in it while a question waited or the copy was made.
            await claim(dest)
            _move_into_place(src, built, dest, replacing)
    except OSError as error:  # shutil.Error is one
        raise PlacementError(f"Could not make {dest} from {src}: {_reason(error)}") from error
    click.secho(f"{'Hardlinked' if linked else 'Copied'} folder to '{dest}'.", fg="yellow")


def _refuse(src: str, dest: str) -> None:
    if (reason := never_replaced(dest) or overlap(dest, src)) is not None:
        raise PlacementDeclined(f"Not putting {src} at {dest}: {reason}.")


def _reason(error: OSError) -> str:
    """What went wrong: for copytree, which collects an error per file, the first and how many more."""
    failures = error.args[0] if isinstance(error, shutil.Error) and error.args else None
    if not isinstance(failures, list) or not failures:
        return str(error)
    more = f" (and {len(failures) - 1} more)" if len(failures) > 1 else ""
    return f"{failures[0][-1]}{more}"


@contextmanager
def _building_in(parent: str) -> Iterator[str]:
    """A folder to build in, removed however the block ends. mkdtemp's is 0700: the album is made inside it."""
    temp = tempfile.mkdtemp(prefix=BUILDING_PREFIX, dir=parent)
    try:
        yield temp
    finally:
        try:
            _remove_folder(temp)
        except OSError as error:
            click.secho(f"Could not remove {temp}: {error}", fg="yellow")


def _remove_folder(path: str) -> None:
    """Remove a folder salmon was building, or one it moved aside to replace. An album's read-only folders are copied
    read-only, so they are made writable first."""
    if not os.path.isdir(path):
        return
    _make_writable(path)
    shutil.rmtree(path)


def _make_writable(folder: str) -> None:
    """Give the owner write permission on folder and each real folder in it, adding to the mode it has.

    Never on a file, which may be a link to another album's, and never through a symlink or junction: os.chmod
    follows one, so it would change the folder it leads to, a library's included.
    """
    if os.path.islink(folder) or os.path.isjunction(folder):
        return
    with suppress(OSError):
        mode = stat.S_IMODE(os.lstat(folder).st_mode)
        if mode & stat.S_IRWXU != stat.S_IRWXU:
            os.chmod(folder, mode | stat.S_IRWXU)
    try:
        with os.scandir(folder) as entries:
            inside = [entry.path for entry in entries if entry.is_dir(follow_symlinks=False)]
    except OSError:
        return  # The removal says what it can't remove
    for path in inside:
        _make_writable(path)


def _link(src: str, dst: str) -> None:
    try:
        os.link(os.path.realpath(src), dst)  # A link to the file, never to a symlink
    except OSError as error:
        if cannot_link(error):
            raise _LinksUnavailable(error.strerror or str(error)) from error
        raise


async def _build(src: str, built: str, dest: str, ignore: Ignore | None) -> bool:
    """Build the copy of src at built. Returns whether it is hardlinked."""
    if cfg.directory.hardlinks and not _reaches_library(src):
        try:
            if os.stat(src).st_dev != os.stat(os.path.dirname(built)).st_dev:
                raise _LinksUnavailable("they are on different filesystems")
            shutil.copytree(src, built, copy_function=_link, ignore=ignore)
            return True
        except _LinksUnavailable as why:
            # The links made so far go before the question, and before the copy: copy2 onto one is SameFileError.
            _remove_folder(built)
            await _agree_to_copy(src, dest, str(why))
    shutil.copytree(src, built, copy_function=shutil.copy2, ignore=ignore)
    return False


def _reaches_library(src: str) -> bool:
    """Whether a link to a file of src could lead into a library: a tag write through it would change the album."""
    if not cfg.directory.library_dirs:
        return False
    if cfg.directory.protects(src):
        return True
    seen: set[str] = set()
    # As copytree goes: into linked folders too.
    for root, folders, files in os.walk(src, followlinks=True):
        if (real := os.path.realpath(root)) in seen:
            folders.clear()
            continue
        seen.add(real)
        if cfg.directory.is_library_path(root):
            return True
        for name in files:
            file = os.path.join(root, name)
            if os.path.islink(file) and cfg.directory.is_library_path(file):
                return True
    return False


async def _agree_to_copy(src: str, dest: str, why: str) -> None:
    size = 0
    for root, _folders, files in os.walk(src):
        for name in files:
            with suppress(OSError):  # Only the question's figure: the copy reports what it can't read
                size += os.path.getsize(os.path.join(root, name))
    album = f"{src} ({size / 1e6:.0f} MB)"
    if await interaction.assume_defaults():
        click.secho(f"-yyy: copying {album} to {dest}, as it can't be hardlinked there ({why}).", fg="yellow")
        return
    if not await interaction.confirm(
        click.style(f"{album} can't be hardlinked to {dest} ({why}). Copy instead?", fg="magenta", bold=True),
        default=True,
        labels=("Copy", "Don't copy"),
    ):
        raise PlacementDeclined(f"Not copying {src} to {dest}.")


def _move_into_place(src: str, built: str, dest: str, replacing: Replacing | None) -> None:
    """Rename the finished folder to dest and give it src's conversion record, all or nothing: whatever stops either,
    the folder to replace is put back.

    Nothing is awaited from the checks to the rename. Another job or program may have changed dest while a question
    waited or the copy was made, so all of them are made again here.
    """
    _refuse(src, dest)
    if replacing is not None and (change := replacing.changed(dest)) is not None:
        raise PlacementDeclined(f"Not replacing {dest}: {change} while salmon was working.")
    # Imported here: the converter package imports the tagger, which imports this module.
    from salmon.converter.conversions import copy_record

    new = os.lstat(built)
    # A sibling: moving a folder within its parent takes no write permission on the folder itself.
    aside = os.path.join(os.path.dirname(dest), f"{REPLACED_PREFIX}{uuid.uuid4().hex[:12]}")
    try:
        if replacing is not None:
            os.rename(dest, aside)
        # os.rename silently replaces an empty folder on POSIX.
        if os.path.lexists(dest):
            raise PlacementDeclined(f"Not replacing {dest}: it appeared while salmon was working.")
        _move(built, dest)
        # Last, as it changes nothing when it fails: a record left at the name would describe another folder.
        copy_record(src, dest)
    except BaseException:
        # A Ctrl-C too, which can come as a rename returns: what is undone is read from the disk.
        _undo(built, dest, aside, new)
        raise
    if replacing is not None:
        _remove_replaced(dest, aside)


def _move(folder: str, to: str) -> None:
    """Rename a folder into another parent, which takes write permission on the folder itself."""
    mode = stat.S_IMODE(os.lstat(folder).st_mode)
    read_only = not mode & stat.S_IWUSR
    if read_only:
        os.chmod(folder, mode | stat.S_IWUSR)
    try:
        os.rename(folder, to)
    finally:
        if read_only:
            with suppress(OSError):
                os.chmod(folder if os.path.lexists(folder) else to, mode)


def _undo(built: str, dest: str, aside: str, new: os.stat_result) -> None:
    """Take the new folder out of dest and put back the one moved aside, as far as either happened."""
    try:
        if os.path.lexists(dest) and os.path.samestat(os.lstat(dest), new):
            _move(dest, built)  # Removed with the folder it was built in
        if not os.path.lexists(aside):
            return
        if not os.path.lexists(dest):
            os.rename(aside, dest)
            return
    except OSError as error:
        click.secho(f"Could not undo the placement at {dest}: {error}", fg="red")
    if os.path.lexists(aside):
        click.secho(f"The folder that was at {dest} is kept, whole, at {aside}: move it back.", fg="red", bold=True)


def _remove_replaced(dest: str, aside: str) -> None:
    try:
        _remove_folder(aside)
    except OSError as error:
        click.secho(
            f"Replaced {dest}, but could not remove all of the folder that was there. What is left is at {aside}: "
            f"remove it yourself ({_reason(error)}).",
            fg="yellow",
        )


class PlacedFolders:
    """What one run placed, and what it must never replace."""

    def __init__(self, *keep: str) -> None:
        """
        Args:
            keep: The folders the run works from: its source, the album given, the first folder.
        """
        self._keep = keep
        self._declined: list[str] = []  # The destinations declined for a folder the run works from
        self._made: dict[str, str] = {}  # Each folder the run made, and the one it was made from
        self._conversion_dirs: dict[tuple[str, str], str] = {}

    def keep(self, *folders: str) -> None:
        """Add folders the run works from, once it knows them."""
        self._keep += folders

    def skipped_over(self, folder: str) -> str | None:
        """A destination declined because it is folder or holds it, or None."""
        return next((dest for dest in self._declined if holds(dest, folder)), None)

    async def place(self, src: str, code: str) -> str:
        """The folder tracker `code` uploads and seeds src from, made if it is not there.

        Raises:
            PlacementDeclined: It was declined, or it must never be replaced.
            PlacementError: It could not be made.
        """
        if (dest := destination(src, code)) is None:
            return src
        if dryrun.active():
            if os.path.lexists(dest):
                dryrun.say(
                    f"{dest} exists: a real run would ask before replacing it, and never replaces the folder it "
                    "started from."
                )
            return src
        for made, made_from in self._made.items():
            if _same_folder(made, dest):
                if made_from == src:
                    # The same string each time: the seedbox tasks and remove_local_after_copy go by it.
                    return made
                raise PlacementDeclined(f"Not replacing {dest}: this run made it, from another folder.")
        try:
            await self._make(src, dest)
        except OSError as error:
            raise PlacementError(f"Could not make {dest} from {src}: {error}") from error
        return dest

    async def _make(self, src: str, dest: str) -> None:
        if (reason := never_replaced(dest)) is not None:
            raise PlacementDeclined(f"Not using {dest}: {reason}.")
        if (reason := overlap(dest, src, *self._keep, *self._made)) is not None:
            self._declined.append(dest)
            raise PlacementDeclined(f"Not using {dest}: {reason}.", kept=True)
        await claim(dest)
        replacing = await _agree_to_replace(dest) if os.path.lexists(dest) else None
        await place_folder(src, dest, replacing=replacing)
        self._made[dest] = src

    def conversion_dir(self, album: str, kind: str, code: str) -> str:
        """Where tracker `code`'s conversion of album is made: where this run made that kind, else its directory."""
        if not cfg.per_tracker_directories():
            return cfg.directory.download_directory
        # Without trailing separators, as made_conversion's dirname gives it to the later trackers: the seedbox tasks
        # and remove_local_after_copy go by the conversion's path string.
        own = os.path.dirname(os.path.join(cfg.download_directory_for(code), ""))
        return self._conversion_dirs.get((album, kind)) or own

    def made_conversion(self, album: str, kind: str, folder: str) -> None:
        """Note a conversion the run made, which later trackers are placed from and which is never replaced."""
        if not cfg.per_tracker_directories():
            return
        self._made[folder] = folder
        self._conversion_dirs.setdefault((album, kind), os.path.dirname(folder))


async def _agree_to_replace(dest: str) -> Replacing:
    """Ask whether to replace the folder at dest (R11: never merged), and give it as it was before the question."""
    # Checked first: confirm does not answer by itself under -yyy.
    if await interaction.assume_defaults():
        raise PlacementDeclined(f"Not replacing {dest} with -yyy (upload.yes_all): it exists already.")
    replacing = Replacing.of(dest)
    if not await interaction.confirm(
        click.style(
            f"A folder already exists at '{dest}'{SHARED_NOTE if replacing.shared else ''}, "
            "would you like to replace it?",
            fg="magenta",
            bold=True,
        ),
        default=False,
        labels=("Replace", "Keep it"),
    ):
        raise PlacementDeclined(f"Not replacing {dest}.")
    return replacing
