# Third-party notices

The Windows installer bundles FFmpeg 9.0.2 Essentials for Windows from
[Gyan Doshi's FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/).

FFmpeg is a separate executable launched by this application. The bundled
build is licensed under GPLv3. Its source is available from the
[FFmpeg source repository](https://github.com/FFmpeg/FFmpeg/tree/946fcce07b)
and the build distribution is documented by
[Gyan Doshi](https://www.gyan.dev/ffmpeg/builds/). The installer includes the
upstream GPL license file.

The build archive used for the installer is
`ffmpeg-9.0.2-essentials_build.zip`, SHA-256:
`60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`.

The application source code is not linked against FFmpeg libraries.

The face-tracking feature uses the YuNet face-detection model from the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet),
licensed under MIT. The model file is
`backend/models/face_detection_yunet_2023mar.onnx`, SHA-256:
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
The corresponding license is included beside the model.

The feature runs locally on CPU using the pinned
`opencv-python-headless` package. Video frames and face detections are not
uploaded to a third-party service.
