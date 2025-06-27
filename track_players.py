!pip install ultralytics opencv-python-headless deep_sort_realtime gdown --quiet

import cv2, csv, torch, gdown
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab import drive, files
from collections import defaultdict

drive.mount('/content/drive')
gdown.download("https://drive.google.com/uc?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD", "best.pt", fuzzy=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("best.pt").to(device)
if device == 'cuda':
    model.half()

tracker = DeepSort(max_age=20, n_init=2, max_iou_distance=0.5)

video_path = "/content/drive/MyDrive/15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("final_output_fixed.mp4", fourcc, fps, (width, height))

id_counts = defaultdict(int)
frame_detections = defaultdict(list)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        if results.names[cls_id] == "player":
            conf = float(box.conf)
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w * h > 300:
                    detections.append(([x1, y1, w, h], conf, None))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        tid = int(track.track_id)
        l, t, r, b = map(int, track.to_ltrb())
        id_counts[tid] += 1
        frame_detections[frame_id].append((tid, l, t, r, b))

cap.release()


top_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)[:22]
id_map = {old_id: f"Player {i+1}" for i, (old_id, _) in enumerate(top_ids)}


cap = cv2.VideoCapture(video_path)
csv_path = "final_log_fixed.csv"

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "player_number", "x1", "y1", "x2", "y2"])
    frame_id = 0
    debug_saved = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id in frame_detections:
            for tid, l, t, r, b in frame_detections[frame_id]:
                label = id_map[tid] if tid in id_map else f"ID {tid}"
                color = (0, 255, 0) if tid in id_map else (0, 0, 255)

                # Clamp box within frame
                l, t, r, b = max(0, l), max(0, t), min(width - 1, r), min(height - 1, b)

                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                writer.writerow([frame_id, label, l, t, r, b])

        out.write(frame)

        if not debug_saved:
            cv2.imwrite("debug_sample_fixed.jpg", frame)
            debug_saved = True

cap.release()
out.release()

most_common = max(id_counts.values()) if id_counts else 0
reid_accuracy = (most_common / frame_id) * 100 if frame_id else 0
print(f"Tracking completed.")
print(f"Re-ID Stability: {reid_accuracy:.2f}%")
print(f"Unique players detected: {len(id_counts)}, Top mapped: {len(top_ids)}")

files.download("final_output_fixed.mp4")
files.download("final_log_fixed.csv")
files.download("debug_sample_fixed.jpg")
