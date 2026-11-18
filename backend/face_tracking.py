from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FaceTrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceCropKeyframe:
    time: float
    x: float
    y: float


@dataclass(frozen=True)
class _Face:
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class _LipTrack:
    face: _Face
    aperture: float
    score: float
    timestamp: float


class _SpeakerActivityTracker:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tracks: list[_LipTrack] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        measurements: list[tuple[_Face, float]],
        timestamp: float,
        speech_active: bool,
    ) -> _Face | None:
        current: list[_LipTrack] = []
        for face, aperture in measurements:
            previous = min(
                (
                    track
                    for track in self.tracks
                    if timestamp - track.timestamp <= 1.0
                ),
                key=lambda track: math.hypot(
                    (track.face.center[0] - face.center[0]) / self.width,
                    (track.face.center[1] - face.center[1]) / self.height,
                ),
                default=None,
            )
            if previous is not None:
                distance = math.hypot(
                    (previous.face.center[0] - face.center[0]) / self.width,
                    (previous.face.center[1] - face.center[1]) / self.height,
                )
            else:
                distance = float("inf")
            elapsed = timestamp - previous.timestamp if previous else 0.0
            score = (
                previous.score * math.exp(-max(elapsed, 0.0) / 0.75)
                if previous
                else 0.0
            )
            if previous is not None and distance <= 0.2 and speech_active:
                score += abs(aperture - previous.aperture)
            current.append(_LipTrack(face, aperture, score, timestamp))
        self.tracks = current
        speaker = max(current, key=lambda track: track.score, default=None)
        return speaker.face if speaker is not None and speaker.score >= 0.02 else None


def _get_lip_landmarker() -> tuple[Any, Any]:
    global _LIP_LANDMARKER
    if _LIP_LANDMARKER is not None:
        return _LIP_LANDMARKER
    model_path = Path(__file__).with_name("models") / _LIP_MODEL_FILENAME
    if not model_path.is_file():
        raise FaceTrackingError(
            "Falta el modelo de lectura labial incluido con la aplicación."
        )
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise FaceTrackingError(
            "Falta MediaPipe para estimar al orador activo. Reinstala la aplicación."
        ) from error
    with _LIP_LANDMARKER_LOCK:
        if _LIP_LANDMARKER is None:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=5,
            )
            _LIP_LANDMARKER = (
                mp,
                vision.FaceLandmarker.create_from_options(options),
            )
        return _LIP_LANDMARKER


def _lip_measurements(
    landmarks_by_face: list[list[Any]],
    faces: list[_Face],
    width: int,
    height: int,
) -> list[tuple[_Face, float]]:
    measurements: list[tuple[_Face, float]] = []
    for landmarks in landmarks_by_face:
        if len(landmarks) <= 291:
            continue
        xs = [float(landmark.x) * width for landmark in landmarks]
        ys = [float(landmark.y) * height for landmark in landmarks]
        landmark_face = _Face(
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
            1.0,
        )
        matched_face = max(
            faces,
            key=lambda face: _intersection_over_union(face, landmark_face),
            default=None,
        )
        if matched_face is None or _intersection_over_union(
            matched_face, landmark_face
        ) == 0:
            continue
        mouth_width = abs(xs[291] - xs[61])
        aperture = abs(ys[14] - ys[13]) / max(mouth_width, 1.0)
        measurements.append((matched_face, aperture))
    return measurements


_SAMPLE_RATE = 4.0
_MAX_DETECTOR_DIMENSION = 1280
_DROPOUT_HOLD_SECONDS = 1.5
_MAX_KEYFRAMES = 48
_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
_LIP_MODEL_FILENAME = "face_landmarker.task"
_LIP_LANDMARKER_LOCK = threading.Lock()
_LIP_LANDMARKER: tuple[Any, Any] | None = None


def _crop_geometry(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[float, float, float]:
    scale = max(output_width / source_width, output_height / source_height)
    max_x = max(0.0, source_width * scale - output_width)
    max_y = max(0.0, source_height * scale - output_height)
    return scale, max_x, max_y


def _crop_position(
    face_center: tuple[float, float],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    headroom: float = 0.42,
) -> tuple[float, float]:
    scale, max_x, max_y = _crop_geometry(
        source_width, source_height, output_width, output_height
    )
    x = min(max_x, max(0.0, face_center[0] * scale - output_width / 2))
    y = min(max_y, max(0.0, face_center[1] * scale - output_height * headroom))
    return x, y


def _intersection_over_union(left: _Face, right: _Face) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


class _SubjectTracker:
    def __init__(self, width: int, height: int, crop_width: float | None = None) -> None:
        self.width = width
        self.height = height
        self.crop_width = crop_width or width * 0.5
        self.target: tuple[float, float] | None = None
        self.face: _Face | None = None
        self.last_seen: float | None = None
        self.last_update: float | None = None

    def reset(self) -> None:
        self.target = None
        self.face = None
        self.last_seen = None
        self.last_update = None

    def _select(self, faces: list[_Face], timestamp: float) -> _Face | None:
        recent = (
            self.face is not None
            and self.last_seen is not None
            and timestamp - self.last_seen <= _DROPOUT_HOLD_SECONDS
        )
        if not recent:
            return max(faces, key=lambda face: face.area * face.confidence, default=None)

        assert self.face is not None
        previous_center = self.face.center

        def distance(face: _Face) -> float:
            center = face.center
            return math.hypot(
                (center[0] - previous_center[0]) / self.width,
                (center[1] - previous_center[1]) / self.height,
            )

        candidate = min(
            faces,
            key=lambda face: distance(face)
            - 0.08 * _intersection_over_union(self.face, face)
            - 0.015 * face.confidence,
        )
        return candidate if distance(candidate) <= 0.28 else None

    def _move_toward(self, desired: tuple[float, float], timestamp: float) -> None:
        if self.target is None:
            self.target = desired
            self.last_update = timestamp
            return

        previous_update = (
            self.last_update if self.last_update is not None else timestamp
        )
        elapsed = max(0.0, timestamp - previous_update)
        if elapsed == 0:
            return
        delta_x = desired[0] - self.target[0]
        delta_y = desired[1] - self.target[1]
        if math.hypot(delta_x, delta_y) < self.width * 0.015:
            self.last_update = timestamp
            return

        response = 1.0 - math.exp(-elapsed / 0.35)
        move_x = delta_x * response
        move_y = delta_y * response
        distance = math.hypot(move_x, move_y)
        max_distance = max(self.crop_width * 0.8, 120.0) * elapsed
        if distance > max_distance:
            move_x *= max_distance / distance
            move_y *= max_distance / distance
        self.target = (self.target[0] + move_x, self.target[1] + move_y)
        self.last_update = timestamp

    def update(
        self,
        faces: list[_Face],
        timestamp: float,
        scene_changed: bool,
        preferred_face: _Face | None = None,
    ) -> tuple[float, float] | None:
        if scene_changed:
            self.reset()

        selected = preferred_face or (self._select(faces, timestamp) if faces else None)
        if selected is not None:
            self.face = selected
            self.last_seen = timestamp
            self._move_toward(selected.center, timestamp)
        elif self.last_seen is not None and timestamp - self.last_seen <= _DROPOUT_HOLD_SECONDS:
            self.last_update = timestamp
        else:
            self._move_toward((self.width / 2, self.height / 2), timestamp)
        return self.target


def _simplify_keyframes(
    points: list[FaceCropKeyframe],
    tolerance: float = 10.0,
) -> list[FaceCropKeyframe]:
    if len(points) <= 2:
        return points

    kept = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        first, last = points[start], points[end]
        duration = last.time - first.time
        if duration <= 0:
            continue
        largest_error = tolerance
        split_at: int | None = None
        for index in range(start + 1, end):
            point = points[index]
            fraction = (point.time - first.time) / duration
            expected_x = first.x + (last.x - first.x) * fraction
            expected_y = first.y + (last.y - first.y) * fraction
            error = max(abs(point.x - expected_x), abs(point.y - expected_y))
            if error > largest_error:
                largest_error = error
                split_at = index
        if split_at is not None:
            kept.add(split_at)
            pending.extend(((start, split_at), (split_at, end)))

    simplified = [points[index] for index in sorted(kept)]
    if len(simplified) <= _MAX_KEYFRAMES:
        return simplified
    return [
        simplified[round(index * (len(simplified) - 1) / (_MAX_KEYFRAMES - 1))]
        for index in range(_MAX_KEYFRAMES)
    ]


def analyze_face_crop_paths(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
) -> list[list[FaceCropKeyframe] | None]:
    model_path = Path(__file__).with_name("models") / _MODEL_FILENAME
    if not model_path.is_file():
        raise FaceTrackingError(
            "Falta el modelo de seguimiento facial incluido con la aplicación."
        )
    try:
        import cv2
    except ImportError as error:
        raise FaceTrackingError(
            "Falta OpenCV para realizar el seguimiento facial. Reinstala la aplicación."
        ) from error

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise FaceTrackingError("No se pudo abrir el video para analizar los rostros.")
    try:
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width < 1 or source_height < 1:
            raise FaceTrackingError("No se pudieron leer las dimensiones del video.")
        scale, max_x, max_y = _crop_geometry(
            source_width, source_height, output_width, output_height
        )
        if max_x < 1 and max_y < 1:
            return [None for _ in intervals]

        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            0.55,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        media_pipe, landmarker = (
            _get_lip_landmarker() if speech_intervals else (None, None)
        )
        tracker = _SubjectTracker(
            source_width,
            source_height,
            output_width / scale,
        )
        speaker_tracker = _SpeakerActivityTracker(source_width, source_height)
        previous_thumbnail = None
        any_face_detected = False
        all_paths: list[list[FaceCropKeyframe] | None] = []

        for interval_start, interval_end in intervals:
            interval_duration = interval_end - interval_start
            sample_count = max(1, math.ceil(interval_duration * _SAMPLE_RATE))
            points: list[FaceCropKeyframe] = []
            for sample_index in range(sample_count):
                local_time = interval_start + interval_duration * sample_index / sample_count
                absolute_time = clip_start + local_time
                capture.set(cv2.CAP_PROP_POS_MSEC, absolute_time * 1000)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise FaceTrackingError(
                        f"No se pudo leer el video en el segundo {absolute_time:.2f}."
                    )

                frame_height, frame_width = frame.shape[:2]
                if frame_width != source_width or frame_height != source_height:
                    raise FaceTrackingError(
                        "Las dimensiones del video cambiaron durante el seguimiento facial."
                    )
                detector_scale = min(
                    1.0,
                    _MAX_DETECTOR_DIMENSION / max(frame_width, frame_height),
                )
                if detector_scale < 1:
                    detector_frame = cv2.resize(
                        frame,
                        (
                            round(frame_width * detector_scale),
                            round(frame_height * detector_scale),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    detector.setInputSize(
                        (detector_frame.shape[1], detector_frame.shape[0])
                    )
                else:
                    detector_frame = frame
                detector.setInputSize(
                    (detector_frame.shape[1], detector_frame.shape[0])
                )
                thumbnail = cv2.cvtColor(
                    cv2.resize(frame, (48, 27)), cv2.COLOR_BGR2GRAY
                )
                scene_changed = (
                    previous_thumbnail is not None
                    and float(cv2.absdiff(previous_thumbnail, thumbnail).mean()) >= 52.0
                )
                previous_thumbnail = thumbnail
                _, detections = detector.detect(detector_frame)
                faces = (
                    [
                        _Face(
                            float(detection[0]) / detector_scale,
                            float(detection[1]) / detector_scale,
                            float(detection[2]) / detector_scale,
                            float(detection[3]) / detector_scale,
                            float(detection[-1]),
                        )
                        for detection in detections
                    ]
                    if detections is not None
                    else []
                )
                any_face_detected |= bool(faces)
                preferred_face = None
                if scene_changed:
                    speaker_tracker.reset()
                if landmarker is not None and media_pipe is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = media_pipe.Image(
                        image_format=media_pipe.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    with _LIP_LANDMARKER_LOCK:
                        result = landmarker.detect(image)
                    landmarks_by_face = result.face_landmarks or []
                    measurements = _lip_measurements(
                        landmarks_by_face,
                        faces,
                        frame_width,
                        frame_height,
                    )
                    speech_active = any(
                        start <= absolute_time <= end
                        for start, end in speech_intervals or []
                    )
                    preferred_face = speaker_tracker.update(
                        measurements,
                        absolute_time,
                        speech_active,
                    )
                target = tracker.update(
                    faces,
                    absolute_time,
                    scene_changed,
                    preferred_face=preferred_face,
                )
                if target is not None:
                    x, y = _crop_position(
                        target, source_width, source_height, output_width, output_height
                    )
                    point_time = local_time - interval_start
                    if scene_changed and points:
                        previous = points[-1]
                        hold_time = max(previous.time, point_time - 0.02)
                        if hold_time > previous.time:
                            points.append(
                                FaceCropKeyframe(hold_time, previous.x, previous.y)
                            )
                    points.append(
                        FaceCropKeyframe(point_time, x, y)
                    )

            if points:
                if points[0].time > 0:
                    points.insert(
                        0,
                        FaceCropKeyframe(0.0, max_x / 2, max_y / 2),
                    )
                if points[-1].time < interval_duration:
                    last = points[-1]
                    points.append(
                        FaceCropKeyframe(interval_duration, last.x, last.y)
                    )
                all_paths.append(_simplify_keyframes(points))
            else:
                all_paths.append(None)

        return all_paths if any_face_detected else [None for _ in intervals]
    except cv2.error as error:
        raise FaceTrackingError(f"El detector facial no pudo analizar el video: {error}") from error
    finally:
        capture.release()
