"""Infer results bus: detections in POSIX shared memory (MediaBus plane).

The infer service ``InferSlot.create(camera_id)`` after each
predict. Downstream plugins ``InferSlot.attach(camera_id)`` — no detector in the
plugin process.

Pairs with ``IpcSlot`` (frames) under the same ``camera_id``. Sync via
``InferPacket.frame_seq`` ≈ ``IpcSlot.seq``.

"""

from __future__ import annotations

import contextlib
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

_NAME_BYTES = 64
_MODEL_ID_BYTES = 32
_MAX_DET_DEFAULT = 300
# infer_seq, frame_seq, width, height, n, max_det, model_id, front, back
_HEADER = struct.Struct(f"<QQiiii{_MODEL_ID_BYTES}s{_NAME_BYTES}s{_NAME_BYTES}s")
# x1,y1,x2,y2,conf,class_id
_DET = struct.Struct("<5fi")
_DET_SIZE = _DET.size

# One row: x1, y1, x2, y2, confidence, class_id
DetRow = tuple[float, float, float, float, float, int]


@dataclass(frozen=True)
class InferPacket:
    """One published infer result on the MediaBus."""

    infer_seq: int
    frame_seq: int
    width: int
    height: int
    model_id: str
    rows: list[DetProp]



class InferSlot:
    """Ping-pong shared-memory bus for detection rows (CPU MediaBus).

    The infer service **creates** the bus. Plugins **attach**. Same ``camera_id`` as
    ``IpcSlot`` (e.g. ``cam0``).
    """

    def __init__(self, camera_id: str | None = None, *, max_det: int = _MAX_DET_DEFAULT) -> None:
        cid = camera_id or f"anon{os.getpid()}{time.time_ns() % 1_000_000_000}"
        self._open(cid, owner=True, max_det=max_det)

    @classmethod
    def create(cls, camera_id: str, *, max_det: int = _MAX_DET_DEFAULT) -> InferSlot:
        """Service side: create (or replace) the infer bus for ``camera_id``."""
        return cls(camera_id, max_det=max_det)

    @classmethod
    def attach(cls, camera_id: str) -> InferSlot:
        """Plugin side: map an existing infer bus. Topic 02 must be running."""
        slot = object.__new__(cls)
        slot._open(camera_id, owner=False, max_det=_MAX_DET_DEFAULT)
        return slot

    def _open(self, camera_id: str, *, owner: bool, max_det: int) -> None:
        self.camera_id = camera_id
        self.owner = owner
        self.max_det = int(max_det)
        self._front: shared_memory.SharedMemory | None = None
        self._back: shared_memory.SharedMemory | None = None
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
            self._write_header(0, 0, 0, 0, 0, self.max_det, "", "", "")
            self._ensure_buffers()
        else:
            try:
                self._meta = shared_memory.SharedMemory(name=name)
                _track_shm(self._meta)  # plugin exit must not unlink the infer service's bus
            except FileNotFoundError as exc:
                self._lock_file.close()
                raise FileNotFoundError(
                    f"Infer bus {camera_id!r} is not running. Start the infer service first, e.g.\n"
                    f"  python topics/02-inference-runtime/run.py --ipc {camera_id}"
                ) from exc
            _seq, _fs, _w, _h, _n, max_det_hdr, _mid, _f, _b = self._read_header()
            self.max_det = int(max_det_hdr) if max_det_hdr > 0 else _MAX_DET_DEFAULT

    @property
    def seq(self) -> int:
        """Infer publishes so far (shared across processes)."""
        return int(self._read_header()[0])

    def publish(
        self,
        frame_seq: int,
        detections: list[DetProp] | list[Any],
        *,
        width: int,
        height: int,
        model_id: str = "det0",
    ) -> None:
        """Copy detections into shm and bump ``seq``."""
        if not self.owner:
            raise RuntimeError("only the InferSlot creator may publish")
        rows = _as_rows(detections)
        self._lock()
        try:
            self._ensure_buffers()
            assert self._back is not None
            n = min(len(rows), self.max_det)
            buf = self._back.buf
            for i in range(n):
                x1, y1, x2, y2, conf, class_id = rows[i]
                _DET.pack_into(
                    buf,
                    i * _DET_SIZE,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    int(class_id),
                )
            self._front, self._back = self._back, self._front
            front_name = self._front.name if self._front is not None else ""
            back_name = self._back.name if self._back is not None else ""
            infer_seq, *_rest = self._read_header()
            self._write_header(
                infer_seq + 1,
                int(frame_seq),
                int(width),
                int(height),
                n,
                self.max_det,
                model_id,
                front_name,
                back_name,
            )
        finally:
            self._unlock()

    def latest(self) -> InferPacket | None:
        """Decode the last published packet, or ``None`` if empty."""
        self._lock()
        try:
            infer_seq, frame_seq, width, height, n, max_det, model_id, front, _back = (
                self._read_header()
            )
            if infer_seq <= 0 or not front:
                return None
            self.max_det = int(max_det) if max_det > 0 else self.max_det
            shm = self._map_named(front)
            if shm is None:
                return None
            n = max(0, min(int(n), self.max_det))
            rows: list[DetProp] = []
            for i in range(n):
                x1, y1, x2, y2, conf, class_id = _DET.unpack_from(shm.buf, i * _DET_SIZE)
                rows.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(class_id)))
            return InferPacket(
                infer_seq=int(infer_seq),
                frame_seq=int(frame_seq),
                width=int(width),
                height=int(height),
                model_id=model_id,
                rows=rows,
            )
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
        """Unmap this process. The creator also unlinks OS objects."""
        names: list[str] = []
        try:
            _s, _fs, _w, _h, _n, _md, _mid, front, back = self._read_header()
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

    def _ensure_buffers(self) -> None:
        nbytes = self.max_det * _DET_SIZE
        if self._back is None:
            self._back = shared_memory.SharedMemory(create=True, size=nbytes)
            _track_shm(self._back)
        if self._front is None:
            self._front = shared_memory.SharedMemory(create=True, size=nbytes)
            _track_shm(self._front)

    def _map_named(self, name: str) -> shared_memory.SharedMemory | None:
        if not name:
            return None
        if self._front is not None and self._front.name == name:
            return self._front
        if self._back is not None and self._back.name == name:
            return self._back
        try:
            shm = shared_memory.SharedMemory(name=name)
            _track_shm(shm)
        except FileNotFoundError:
            return None
        if self._front is not None:
            self._front.close()
        self._front = shm
        return shm

    def _read_header(
        self,
    ) -> tuple[int, int, int, int, int, int, str, str, str]:
        (
            infer_seq,
            frame_seq,
            width,
            height,
            n,
            max_det,
            model_id,
            front,
            back,
        ) = _HEADER.unpack_from(self._meta.buf)
        return (
            infer_seq,
            frame_seq,
            width,
            height,
            n,
            max_det,
            _unpack_str(model_id),
            _unpack_str(front),
            _unpack_str(back),
        )

    def _write_header(
        self,
        infer_seq: int,
        frame_seq: int,
        width: int,
        height: int,
        n: int,
        max_det: int,
        model_id: str,
        front: str,
        back: str,
    ) -> None:
        _HEADER.pack_into(
            self._meta.buf,
            0,
            infer_seq,
            frame_seq,
            width,
            height,
            n,
            max_det,
            _pack_str(model_id, _MODEL_ID_BYTES),
            _pack_str(front, _NAME_BYTES),
            _pack_str(back, _NAME_BYTES),
        )

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
    """POSIX shm name for the infer bus header."""
    return f"cv_infer_{_safe_id(camera_id)}"


def _as_rows(detections: list[DetProp] | list[Any]) -> list[DetProp]:
    if not detections:
        return []
    first = detections[0]
    if isinstance(first, tuple):
        return list(detections)  # type: ignore[arg-type]
    rows: list[DetProp] = []
    for d in detections:  # type: ignore[assignment]
        x1, y1, x2, y2 = d.bbox
        rows.append((float(x1), float(y1), float(x2), float(y2), float(d.confidence), int(d.class_id)))
    return rows


def _safe_id(camera_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in camera_id)[:48]


def _lock_path(camera_id: str) -> Path:
    return Path(tempfile.gettempdir()) / f"computer-vis-infer-{_safe_id(camera_id)}.lock"


def _pack_str(value: str, size: int) -> bytes:
    return value.encode("ascii", errors="replace")[: size - 1].ljust(size, b"\0")


def _unpack_str(raw: bytes) -> str:
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

    Same rationale as ``common.media.ipc._track_shm``: consumer/plugin exit
    must not unlink buses owned by topic 01 / 02.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except Exception:
        pass
