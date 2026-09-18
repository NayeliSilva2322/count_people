"""CPU IPC: one BGR frame in POSIX shared memory (no pickle).

Threads in one process already share the Python heap. Use this when a
**second process** must see the same pixels.

``IpcFrame`` is one buffer. ``IpcSlot`` is the ping-pong pair plus seq/wait
used by producer and consumer processes.

"""

from __future__ import annotations

import contextlib
import os
import struct
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

_NAME_BYTES = 64
# seq, width, height, front shm name, back shm name
_HEADER = struct.Struct("<Qii64s64s")


class IpcFrame:
    """One BGR ``uint8`` image in POSIX shared memory (CPU IPC)."""

    def __init__(
        self,
        height: int,
        width: int,
        shm: shared_memory.SharedMemory,
        *,
        owner: bool,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.shm = shm
        self.owner = owner
        self.array = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=shm.buf)

    @classmethod
    def create(cls, height: int, width: int, name: str | None = None) -> IpcFrame:
        """Allocate a new IPC buffer. Caller must ``close()`` / ``unlink()``."""
        nbytes = int(height) * int(width) * 3
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        _track_shm(shm)
        frame = cls(height, width, shm, owner=True)
        frame.array[:] = 0
        return frame

    @classmethod
    def attach(cls, name: str, height: int, width: int) -> IpcFrame:
        """Map an existing buffer from another process (same ``name``)."""
        shm = shared_memory.SharedMemory(name=name)
        _track_shm(shm)  # consumer exit must not unlink the producer's bus
        return cls(height, width, shm, owner=False)

    @property
    def name(self) -> str:
        """OS name other processes pass to ``attach``."""
        return str(self.shm.name)

    def close(self) -> None:
        """Unmap this process. Does not delete the OS object."""
        self.shm.close()

    def unlink(self) -> None:
        """Delete the OS object. Only the creator should call this."""
        if self.owner:
            self.shm.unlink()


class IpcSlot:
    """Ping-pong pair of ``IpcFrame`` buffers (CPU IPC).

    The capture service **creates** the bus (``IpcSlot.create("cam0")``). Downstream consumers
    **attaches** (``IpcSlot.attach("cam0")``) from another process — no pickle,
    no common parent. Do not write in place on ``latest()``.
    """

    def __init__(self, camera_id: str | None = None) -> None:
        """Create a new bus. ``camera_id`` is the public name (default: unique)."""
        cid = camera_id or f"anon{os.getpid()}{time.time_ns() % 1_000_000_000}"
        self._open(cid, owner=True)

    @classmethod
    def create(cls, camera_id: str) -> IpcSlot:
        """Producer side: create (or replace) the named bus for ``camera_id``."""
        return cls(camera_id)

    @classmethod
    def attach(cls, camera_id: str) -> IpcSlot:
        """Consumer side: map an existing bus. Topic 01 producer must be running."""
        slot = object.__new__(cls)
        slot._open(camera_id, owner=False)
        return slot

    def _open(self, camera_id: str, *, owner: bool) -> None:
        self.camera_id = camera_id
        self.owner = owner
        self._front: IpcFrame | None = None
        self._back: IpcFrame | None = None
        lock_path = _lock_path(camera_id)
        lock_path.touch(exist_ok=True)
        self._lock_file = lock_path.open("a+b")
        self._lock_file.seek(0, os.SEEK_END)
        if self._lock_file.tell() == 0:
            self._lock_file.write(b"\0")
            self._lock_file.flush()
        name = meta_shm_name(camera_id)
        if owner:
            _unlink_shm(name)
            self._meta = shared_memory.SharedMemory(create=True, size=_HEADER.size, name=name)
            _track_shm(self._meta)
            self._write_header(0, 0, 0, "", "")
        else:
            try:
                self._meta = shared_memory.SharedMemory(name=name)
                _track_shm(self._meta)  # attach only — the capture service keeps ownership
            except FileNotFoundError as exc:
                self._lock_file.close()
                raise FileNotFoundError(
                    f"IPC bus {camera_id!r} is not running. Start the capture service first, e.g.\n"
                    f"  python topics/01-capture-runtime/run.py --source 0 --no-consumer"
                ) from exc

    def __getstate__(self) -> dict[str, str]:
        """Spawn workers attach by name. Do not pickle file handles."""
        return {"camera_id": self.camera_id}

    def __setstate__(self, state: dict[str, str]) -> None:
        self._open(str(state["camera_id"]), owner=False)

    @property
    def seq(self) -> int:
        """Frames published so far (shared across processes)."""
        return int(self._read_header()[0])

    def publish(self, frame: np.ndarray) -> None:
        """Copy ``frame`` into CPU IPC memory and make it the readable front."""
        height, width = frame.shape[:2]
        self._lock()
        try:
            if self._back is None or self._back.height != height or self._back.width != width:
                self._replace_buffers(height, width)
            assert self._back is not None
            self._back.array[:] = frame
            self._front, self._back = self._back, self._front
            seq, _w, _h, _front, _back = self._read_header()
            self._write_header(
                seq + 1,
                width,
                height,
                self._front.name if self._front is not None else "",
                self._back.name if self._back is not None else "",
            )
        finally:
            self._unlock()

    def latest(self) -> np.ndarray | None:
        """Zero-copy view of the last published frame, or ``None``."""
        self._lock()
        try:
            self._map_front()
            if self._front is None:
                return None
            return self._front.array
        finally:
            self._unlock()

    def wait(self, seq: int, timeout_s: float) -> bool:
        """Block until ``self.seq`` is no longer ``seq`` (or timeout)."""
        if timeout_s <= 0:
            return self.seq != seq
        deadline = time.monotonic() + timeout_s
        while self.seq == seq:
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return True

    def close(self) -> None:
        """Unmap this process. The creator also unlinks the OS objects."""
        names = []
        try:
            _seq, _w, _h, front, back = self._read_header()
            names.extend((front, back))
        except Exception:
            pass
        for buf in (self._front, self._back):
            if buf is None:
                continue
            names.append(buf.name)
            buf.close()
        self._front = None
        self._back = None
        meta_name = self._meta.name
        self._meta.close()
        if self.owner:
            seen: set[str] = set()
            for name in names:
                if name and name not in seen:
                    seen.add(name)
                    _unlink_shm(name)
            _unlink_shm(meta_name)
            with contextlib.suppress(OSError):
                Path(self._lock_file.name).unlink(missing_ok=True)
        self._lock_file.close()

    def _replace_buffers(self, height: int, width: int) -> None:
        old = []
        for buf in (self._front, self._back):
            if buf is None:
                continue
            old.append(buf.name)
            buf.close()
        self._front = IpcFrame.create(height, width)
        self._back = IpcFrame.create(height, width)
        for name in old:
            _unlink_shm(name)
        self._write_header(self.seq, width, height, self._front.name, self._back.name)

    def _map_front(self) -> None:
        _seq, width, height, name, _back = self._read_header()
        if not name or width <= 0 or height <= 0:
            return
        if (
            self._front is not None
            and self._front.name == name
            and self._front.width == width
            and self._front.height == height
        ):
            return
        if self._front is not None:
            self._front.close()
            self._front = None
        self._front = IpcFrame.attach(name, height, width)

    def _read_header(self) -> tuple[int, int, int, str, str]:
        seq, width, height, front, back = _HEADER.unpack_from(self._meta.buf)
        return seq, width, height, _unpack_name(front), _unpack_name(back)

    def _write_header(self, seq: int, width: int, height: int, front: str, back: str) -> None:
        _HEADER.pack_into(self._meta.buf, 0, seq, width, height, _pack_name(front), _pack_name(back))

    def _lock(self) -> None:
        fd = self._lock_file.fileno()
        if os.name == "nt":
            import msvcrt

            self._lock_file.seek(0)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(self) -> None:
        fd = self._lock_file.fileno()
        if os.name == "nt":
            import msvcrt

            self._lock_file.seek(0)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def meta_shm_name(camera_id: str) -> str:
    """POSIX shm name for the bus header (producer create / consumer attach)."""
    return f"cv_ipc_{_safe_id(camera_id)}"


def _safe_id(camera_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in camera_id)[:48]


def _lock_path(camera_id: str) -> Path:
    return Path(tempfile.gettempdir()) / f"computer-vis-ipc-{_safe_id(camera_id)}.lock"


def _pack_name(value: str) -> bytes:
    return value.encode("ascii", errors="replace")[: _NAME_BYTES - 1].ljust(_NAME_BYTES, b"\0")


def _unpack_name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def _unlink_shm(name: str) -> None:
    if not name:
        return
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _track_shm(shm: shared_memory.SharedMemory) -> None:
    """Unregister from CPython ``resource_tracker`` after create/attach.

    Otherwise the tracker unlinks the POSIX name when *any* process that
    opened the segment exits — killing topic 02/03 would delete the capture service's
    frame bus (``(deleted)`` in ``/proc``) while the producer keeps writing.
    Only the owner ``close()`` path may unlink.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except Exception:
        pass
