"""What placement checks of a folder before it replaces and deletes it (#1056). It only reads."""

import os
import re
import stat

from salmon.containment import is_within

# An entry's path in the folder, and what tells a change of it.
Entry = tuple[str, tuple[int, ...]]
# Linux's table of this process's mounts, the only thing that shows a bind mount of one filesystem.
MOUNTS = "/proc/self/mountinfo"


def folder_state(folder: str) -> frozenset[Entry]:
    """Each entry below folder as it is now, following no link. Two states differ when something in it changed.

    A file's ctime counts only while it has one link: linking the album to place changes that of a file an earlier run
    linked from it. So an edit that keeps the size and mtime of such a file goes unseen: it is in the other link too.
    """
    found: set[Entry] = set()

    def unreadable(error: OSError) -> None:
        found.add((os.path.relpath(error.filename or folder, folder), (error.errno or -1,)))

    for root, folders, files in os.walk(folder, onerror=unreadable):
        for name in (*folders, *files):
            path = os.path.join(root, name)
            try:
                found.add((os.path.relpath(path, folder), _entry_state(os.lstat(path))))
            except OSError as error:
                found.add((os.path.relpath(path, folder), (error.errno or -1,)))
        folders[:] = [name for name in folders if not os.path.isjunction(os.path.join(root, name))]
    return frozenset(found)


def _entry_state(entry: os.stat_result) -> tuple[int, ...]:
    own_ctime = stat.S_ISDIR(entry.st_mode) or entry.st_nlink == 1
    return (
        entry.st_dev,
        entry.st_ino,
        entry.st_mode,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns if own_ctime else 0,
    )


def mount_inside(folder: str) -> str | None:
    """A mount point that is folder or is in it, or None. Removing folder would empty what is mounted there.

    Without Linux's table each folder below is asked os.path.ismount, which misses a bind mount of one filesystem.
    """
    real = os.path.realpath(folder)
    try:
        points = _mount_points()
    except OSError:
        return _ismount_inside(real)
    return next((point for point in points if is_within(point, real)), None)


def _mount_points() -> list[str]:
    with open(MOUNTS, "rb") as table:
        # Split at spaces alone, as a path can hold other white space. One in it is written in octal (\\040).
        fields = [line.split(b" ") for line in table]
    return [os.fsdecode(re.sub(rb"\\([0-7]{3})", lambda code: bytes([int(code[1], 8)]), each[4])) for each in fields]


def _ismount_inside(real: str) -> str | None:
    if os.path.ismount(real):
        return real
    for root, folders, _files in os.walk(real):
        below = [os.path.join(root, name) for name in folders]
        # Never through a link: what that leads to is not removed.
        own = [path for path in below if not (os.path.islink(path) or os.path.isjunction(path))]
        if point := next((path for path in own if os.path.ismount(path)), None):
            return point
        folders[:] = [os.path.basename(path) for path in own]
    return None
