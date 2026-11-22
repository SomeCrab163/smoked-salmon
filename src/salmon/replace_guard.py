"""What placement checks of a folder before it replaces and deletes it (#1056). A leaf module: it only reads."""

import os
import stat

# An entry's path in the folder, and what tells a change of it.
Entry = tuple[str, tuple[int, ...]]


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
