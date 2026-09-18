"""MediaBus IPC subset for deploy plugins (IpcSlot + InferSlot only)."""

from common.media.infer_ipc import InferPacket, InferSlot
from common.media.ipc import IpcFrame, IpcSlot

__all__ = ["InferPacket", "InferSlot", "IpcFrame", "IpcSlot"]
