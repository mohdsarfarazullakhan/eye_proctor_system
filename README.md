# eye_proctor_system
A real-time computer vision project using OpenCV and MediaPipe Face Mesh to detect:

Eye blinks

Eye aspect ratio (EAR)

Iris center

Gaze direction (LEFT / RIGHT / CENTER)

Log events with timestamps

Save logs to CSV

This system can be used for AI exam proctoring, fatigue detection, driver monitoring, or attention tracking.

** Features**
Real-Time Monitoring

Detects eye landmarks using MediaPipe Face Mesh

Calculates Eye Aspect Ratio (EAR)

Identifies blink events


**Gaze Tracking**

Calculates iris center

Determines gaze direction:

LEFT

RIGHT

CENTER

**Logging**

Each blink is recorded

Each off-center gaze is recorded

Auto-saves events to:

eye_events_log.csv

Live Webcam UI

Eye landmarks drawn on screen

EAR displayed

Blink count shown

Gaze direction displayed
