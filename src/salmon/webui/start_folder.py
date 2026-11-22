"""The folder a job was started on, known again once the job runs (#1056)."""

import os
from contextlib import suppress


class StartFolder:
    """A folder as it was when a job was started on it, to tell whether the job still finds that one when it runs.

    The same folder is the same device and inode, whatever changed in it: an earlier job may have retagged it. Those
    alone would take a deleted and remade folder for the old one, as ext4 gives the new one the same inode at once.
    So on Linux the folder stays open (O_PATH: no read, no permission on it needed) until ``close``, which keeps its
    inode from being given out again. Elsewhere its creation time, where there is one, tells them apart.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._held: int | None = None
        self._was: tuple[int, ...] | None = None
        if flags := getattr(os, "O_PATH", 0):
            with suppress(OSError):
                self._held = os.open(path, flags)
        with suppress(OSError):
            self._was = _identity(os.stat(path) if self._held is None else os.fstat(self._held))

    def is_another(self) -> bool:
        """Whether the path no longer leads to that folder. Never for a path that led to none then."""
        if self._was is None:
            return False
        try:
            return _identity(os.stat(self._path)) != self._was
        except OSError:
            return True

    def close(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            with suppress(OSError):
                os.close(held)


def _identity(entry: os.stat_result) -> tuple[int, ...]:
    return entry.st_dev, entry.st_ino, getattr(entry, "st_birthtime_ns", 0)
