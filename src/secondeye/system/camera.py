"""Non-blocking camera capture and latest-result inference runtime."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    captured_at: float
    frame: Any


class LatestFrameBuffer:
    """A queue of size one: new camera frames replace stale unprocessed frames."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._condition = threading.Condition()
        self._packet: FramePacket | None = None
        self._closed = False
        self._error: Exception | None = None

    def publish(self, frame: Any, captured_at: float | None = None) -> FramePacket:
        if frame is None:
            raise ValueError("frame không được là None")
        with self._condition:
            next_id = 0 if self._packet is None else self._packet.frame_id + 1
            packet = FramePacket(
                frame_id=next_id,
                captured_at=self.clock() if captured_at is None else captured_at,
                frame=frame,
            )
            self._packet = packet
            self._condition.notify_all()
            return packet

    def latest(self, *, copy_frame: bool = True) -> FramePacket | None:
        with self._condition:
            packet = self._packet
            if packet is None:
                return None
            frame = packet.frame.copy() if copy_frame else packet.frame
            return FramePacket(packet.frame_id, packet.captured_at, frame)

    def wait_for_new(
        self, after_frame_id: int, timeout: float = 0.5
    ) -> FramePacket | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed
                or self._error is not None
                or (
                    self._packet is not None and self._packet.frame_id > after_frame_id
                ),
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(
                    f"Camera capture lỗi: {self._error}"
                ) from self._error
            if self._packet is None or self._packet.frame_id <= after_frame_id:
                return None
            packet = self._packet
            return FramePacket(packet.frame_id, packet.captured_at, packet.frame.copy())

    def close(self, error: Exception | None = None) -> None:
        with self._condition:
            self._closed = True
            self._error = error
            self._condition.notify_all()


class LatestFrameCapture:
    """Continuously capture camera frames without blocking inference or UI."""

    def __init__(
        self,
        cv2_module: Any,
        camera_index: int,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
    ) -> None:
        if width <= 0 or height <= 0 or target_fps <= 0:
            raise ValueError("camera width/height/fps phải dương")
        self.cv2 = cv2_module
        self.camera_index = camera_index
        # Let OpenCV choose its default backend. Explicit CAP_AVFOUNDATION can
        # open a Continuity Camera but still fail to deliver its first frames.
        self.capture = cv2_module.VideoCapture(camera_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"Không mở được camera {camera_index}. Kiểm tra quyền Camera và Continuity Camera."
            )
        self.capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, float(width))
        self.capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, float(height))
        self.capture.set(cv2_module.CAP_PROP_FPS, float(target_fps))
        if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1.0)
        self.frames = LatestFrameBuffer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.measured_fps = 0.0

    def start(self) -> "LatestFrameCapture":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="secondeye-camera-capture", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        previous = None
        failure_started = None
        successful_frames = 0
        try:
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    failure_started = failure_started or time.monotonic()
                    grace_seconds = 5.0 if successful_frames == 0 else 1.0
                    if time.monotonic() - failure_started <= grace_seconds:
                        self._stop.wait(0.05)
                        continue
                    raise RuntimeError(
                        "camera không trả frame hợp lệ sau thời gian chờ; "
                        "hãy mở khóa iPhone và bật Continuity Camera"
                    )
                failure_started = None
                successful_frames += 1
                captured_at = time.monotonic()
                if previous is not None and captured_at > previous:
                    instant = 1.0 / (captured_at - previous)
                    self.measured_fps = (
                        instant
                        if self.measured_fps == 0.0
                        else 0.9 * self.measured_fps + 0.1 * instant
                    )
                previous = captured_at
                self.frames.publish(frame, captured_at)
        except Exception as exc:  # surfaced to UI thread via LatestFrameBuffer
            self.frames.close(exc)
        else:
            self.frames.close()

    def stop(self) -> None:
        self._stop.set()
        self.frames.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            # Release after the regular read loop exits. Releasing AVFoundation
            # concurrently with capture.read() can terminate the Python process.
            self.capture.release()
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None
        else:
            self.capture.release()


class AsyncVisionRuntime:
    """Run slow models off the UI thread and always consume the newest frame."""

    def __init__(
        self,
        system: Any,
        frames: LatestFrameBuffer,
        *,
        detection_fps: float = 12.0,
        depth_fps: float = 3.0,
        max_depth_age_seconds: float = 0.50,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if detection_fps <= 0 or depth_fps <= 0:
            raise ValueError("detection_fps và depth_fps phải dương")
        if max_depth_age_seconds <= 0:
            raise ValueError("max_depth_age_seconds phải dương")
        self.system = system
        self.frames = frames
        self.detection_interval = 1.0 / detection_fps
        self.depth_interval = 1.0 / depth_fps
        self.max_depth_age_seconds = max_depth_age_seconds
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, object] | None = None
        self._error: Exception | None = None
        self.measured_detection_fps = 0.0
        self.measured_depth_fps = 0.0

    def start(self) -> "AsyncVisionRuntime":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="secondeye-vision-worker", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        last_frame_id = -1
        last_detection_started = float("-inf")
        last_depth_started = float("-inf")
        latest_depth: dict[str, object] | None = None
        latest_depth_at: float | None = None
        latest_depth_frame_id: int | None = None
        previous_detection_completed: float | None = None
        previous_depth_completed: float | None = None
        try:
            while not self._stop.is_set():
                packet = self.frames.wait_for_new(last_frame_id, timeout=0.25)
                if packet is None:
                    continue
                last_frame_id = packet.frame_id
                now = self.clock()
                wait_seconds = self.detection_interval - (now - last_detection_started)
                if wait_seconds > 0 and self._stop.wait(wait_seconds):
                    break
                packet = self.frames.latest(copy_frame=True) or packet
                last_frame_id = packet.frame_id
                started = self.clock()
                last_detection_started = started

                if (
                    self.system.depth is not None
                    and started - last_depth_started >= self.depth_interval
                ):
                    last_depth_started = started
                    latest_depth = self.system.depth.predict_bgr(packet.frame)
                    # Age from the source frame capture time, not from model
                    # completion, so stale depth cannot be treated as fresh.
                    latest_depth_at = packet.captured_at
                    latest_depth_frame_id = packet.frame_id
                    depth_completed = self.clock()
                    if (
                        previous_depth_completed is not None
                        and depth_completed > previous_depth_completed
                    ):
                        instant = 1.0 / (depth_completed - previous_depth_completed)
                        self.measured_depth_fps = (
                            instant
                            if self.measured_depth_fps == 0.0
                            else 0.8 * self.measured_depth_fps + 0.2 * instant
                        )
                    previous_depth_completed = depth_completed

                detection = self.system.detector.predict_bgr(packet.frame)
                completed = self.clock()
                if (
                    previous_detection_completed is not None
                    and completed > previous_detection_completed
                ):
                    instant = 1.0 / (completed - previous_detection_completed)
                    self.measured_detection_fps = (
                        instant
                        if self.measured_detection_fps == 0.0
                        else 0.8 * self.measured_detection_fps + 0.2 * instant
                    )
                previous_detection_completed = completed
                depth_age = (
                    None
                    if latest_depth_at is None
                    else max(0.0, completed - latest_depth_at)
                )
                usable_depth = (
                    latest_depth
                    if (
                        depth_age is not None
                        and depth_age <= self.max_depth_age_seconds
                        and latest_depth_frame_id == packet.frame_id
                    )
                    else None
                )
                payload = self.system.fuse_detection_and_depth(
                    detection,
                    usable_depth,
                    started_at=started,
                    depth_age_ms=None if depth_age is None else depth_age * 1000.0,
                )
                payload.update(
                    {
                        "frame_id": packet.frame_id,
                        "captured_at": packet.captured_at,
                        "completed_at": completed,
                        "result_age_ms": round(
                            max(0.0, completed - packet.captured_at) * 1000.0, 2
                        ),
                        "depth_source_frame_id": latest_depth_frame_id,
                        "depth_synchronized": bool(
                            usable_depth is not None
                            and latest_depth_frame_id == packet.frame_id
                        ),
                        "depth_rejection_reason": (
                            None
                            if usable_depth is not None or latest_depth is None
                            else (
                                "stale"
                                if depth_age is not None
                                and depth_age > self.max_depth_age_seconds
                                else "different_frame"
                            )
                        ),
                    }
                )
                with self._lock:
                    self._latest = payload
        except Exception as exc:  # surfaced to UI thread
            with self._lock:
                self._error = exc

    def latest(self) -> dict[str, object] | None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"Vision worker lỗi: {self._error}") from self._error
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
