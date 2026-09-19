import uuid

import numpy as np
import pytest

from common.media import InferSlot, IpcSlot


def test_ipc_slot_attach_round_trip():
    camera_id = f"test-ipc-{uuid.uuid4().hex}"
    producer = IpcSlot.create(camera_id)
    consumer = IpcSlot.attach(camera_id)

    try:
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

        producer.publish(frame)

        latest = consumer.latest()

        assert latest is not None
        assert consumer.seq == 1
        np.testing.assert_array_equal(latest, frame)
    finally:
        consumer.close()
        producer.close()


def test_infer_slot_attach_round_trip():
    camera_id = f"test-infer-{uuid.uuid4().hex}"
    producer = InferSlot.create(camera_id)
    consumer = InferSlot.attach(camera_id)

    try:
        detections = [(1.0, 2.0, 3.0, 4.0, 0.9, 1)]

        producer.publish(frame_seq=7, detections=detections, width=640, height=480, model_id="model")

        latest = consumer.latest()

        assert latest is not None
        assert latest.infer_seq == 1
        assert latest.frame_seq == 7
        assert latest.width == 640
        assert latest.height == 480
        assert latest.model_id == "model"
        assert latest.rows[0][:4] == detections[0][:4]
        assert latest.rows[0][4] == pytest.approx(detections[0][4])
        assert latest.rows[0][5] == detections[0][5]
    finally:
        consumer.close()
        producer.close()
