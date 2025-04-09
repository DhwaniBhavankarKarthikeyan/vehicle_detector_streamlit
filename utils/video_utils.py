import cv2
from ultralytics import YOLO
import tempfile
import os

def detect_vehicles_in_video(video_path, timestamps, output_path=None):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (int(cap.get(3)), int(cap.get(4))))

    frame_indices = [int(fps * t) for t in timestamps]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            results = model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ['car', 'truck']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path
