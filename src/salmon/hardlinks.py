"""What an error from os.link means. A leaf module: it imports nothing of salmon's, so anything can import it."""

import errno

# What os.link raises where the volume has no hard links, or no more for the file.
_NO_HARD_LINKS = frozenset(
    code
    for code in (getattr(errno, name, None) for name in ("EPERM", "ENOTSUP", "EOPNOTSUPP", "EMLINK", "ENOSYS"))
    if code is not None
)
# Windows' ERROR_INVALID_FUNCTION, from a FAT or exFAT volume; its errno, EINVAL, means other things elsewhere.
_WINDOWS_NO_HARD_LINKS = 1


def no_hard_links(error: OSError) -> bool:
    """Whether os.link failed for want of hard links on the volume, or of more for the file."""
    return error.errno in _NO_HARD_LINKS or getattr(error, "winerror", None) == _WINDOWS_NO_HARD_LINKS


def cannot_link(error: OSError) -> bool:
    """no_hard_links, or two paths a link cannot join: another filesystem, or another mount of the same one."""
    return error.errno == errno.EXDEV or no_hard_links(error)
