from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_PATH = "imagepi.jpg"

def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_PATH,
        conf=0.25,
        save=True,
        show=False
    )

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("No detection found.")
            return

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f"Class      : {class_name}")
            print(f"Confidence : {conf:.4f}")
            print(f"Box        : ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print("-" * 40)

if __name__ == "__main__":
    main()