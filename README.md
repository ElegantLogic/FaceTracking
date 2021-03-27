# Real-Time Face Point Tracking

A real-time face point tracker using Kanade–Lucas–Tomasi (KLT) feature tracker. The tracker works as follows:

- Detect faces from the frame using Haar Cascade Detector if no face keypoints
are available.
- Extract a rectangle surrounding the face from the frame, cropping it if
necessary.
- Get trackable points from the face rectangle (good features to track).
- Calculate the optical flow of those points relative to the same points in the
previous frame.
- Draw the location of the new points calculated from their optical flow.
- Repeat for each frame.
