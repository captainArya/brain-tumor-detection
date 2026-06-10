import os
from datetime import datetime
import pickle
import shutil

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from report_generator import generate_report

def load_training_metrics():
    csv_path = "runs/detect/train/results.csv"

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # remove NaN rows (VERY IMPORTANT)
    df = df.dropna()

    last = df.iloc[-1]

    precision = float(last.get("metrics/precision(B)", 0))
    recall = float(last.get("metrics/recall(B)", 0))
    map50 = float(last.get("metrics/mAP50(B)", 0))
    map5095 = float(last.get("metrics/mAP50-95(B)", 0))

    return {
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095
    }
def format_metrics():
    m = load_training_metrics()
    if not m:
        return None

    precision = m["Precision"]
    recall = m["Recall"]

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "Accuracy": m["mAP50"],   # correct mapping for detection
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }    

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Detection System", layout="wide")

# ---------------- CONFIG ----------------
FEEDBACK_FILE = "feedback/doctor_feedback.csv"
UPLOAD_DIR = "uploads"
MEMORY_FILE = "feedback/memory_db.pkl"

DATA_YAML = "feedback_data.yaml"   # already created by you

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("feedback_dataset/images", exist_ok=True)
os.makedirs("feedback_dataset/labels", exist_ok=True)

# ---------------- MEMORY ----------------
def cosine_similarity_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

def get_image_embedding(image):
    image = cv2.resize(image, (224, 224))
    return image.flatten().astype(np.float32) / 255.0

def find_similar_case(current_embedding, memory, threshold=0.92):
    if not memory:
        return None
    sims = [cosine_similarity_np(current_embedding, m["embedding"]) for m in memory]
    best = np.argmax(sims)
    return memory[best] if sims[best] >= threshold else None

# ---------------- SAVE DATASET ----------------
def save_feedback_dataset(image_path, corrected_class):
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}

    img_name = os.path.basename(image_path)
    new_img_path = f"feedback_dataset/images/{img_name}"

    shutil.copy(image_path, new_img_path)

    if "manual_bbox" not in st.session_state:
        st.error("No bounding box found")
        return

    x1, y1, x2, y2 = st.session_state["manual_bbox"]

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    label_path = f"feedback_dataset/labels/{os.path.splitext(img_name)[0]}.txt"

    with open(label_path, "w") as f:
        f.write(f"{class_map[corrected_class]} {x_center} {y_center} {bw} {bh}")

# ---------------- MODEL ----------------
def find_latest_model():
    latest, tmax = None, 0
    for root, _, files in os.walk("runs"):
        if "best.pt" in files:
            p = os.path.join(root, "best.pt")
            t = os.path.getmtime(p)
            if t > tmax:
                latest, tmax = p, t
    return latest

@st.cache_resource
def load_model(path):
    if path is None:
        st.error("❌ No trained model found")
        st.stop()
    return YOLO(path)

MODEL_PATH = find_latest_model()
model = load_model(MODEL_PATH)

# ---------------- HEATMAP ----------------
def generate_heatmap(img, box):
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    x1, y1, x2, y2 = map(int, box)
    heatmap[y1:y2, x1:x2] = 1
    heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
    heatmap /= max(np.max(heatmap), 1e-6)
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ---------------- METRICS ----------------
def evaluate_model(path):
    model_eval = YOLO(path)
    metrics = model_eval.val(data="data/data.yaml", split="val", verbose=False)
    return float(metrics.box.map50)

# ---------------- RETRAIN ----------------
def retrain_model_fast():
    global MODEL_PATH, model

    img_count = len(os.listdir("feedback_dataset/images"))
    lbl_count = len(os.listdir("feedback_dataset/labels"))

    if img_count == 0 or lbl_count == 0:
        st.error("❌ No dataset available for training")
        return

    st.info("🚀 Training (fast fine-tuning)...")

    model_new = YOLO(MODEL_PATH)

    model_new.train(
        data=DATA_YAML,
        epochs=5,
        batch=4,
        freeze=10,
        project="runs/retrain",
        name="feedback",
        exist_ok=True
    )

    new_path = model_new.trainer.best

    old_map = evaluate_model(MODEL_PATH)
    new_map = evaluate_model(new_path)

    st.metric("Old mAP50", old_map)
    st.metric("New mAP50", new_map)
    st.metric("Improvement", new_map - old_map)

    choice = st.radio("Use new model?", ["Yes", "No"])

    if choice == "Yes" and new_map >= old_map:
        MODEL_PATH = new_path
        st.cache_resource.clear()
        model = load_model(MODEL_PATH)
        st.success("✅ New model deployed")
    else:
        st.warning("⚠️ Old model kept")

# ---------------- COUNT ----------------
def get_corrected_count():
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        return len(df[df["decision"] == "Corrected"])
    return 0

# ---------------- UI ----------------
st.title("Brain Tumor Detection with Explainable AI and Human Feedback")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Upload", "Detection", "Feedback", "History", "Improvement"]
)

# -------- Upload --------
with tab1:
    file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])
    if file:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.session_state["img"] = img
        st.session_state["img_rgb"] = img_rgb
        st.session_state["path"] = path

        st.image(img_rgb)

# -------- Detection --------
with tab2:
    if "path" in st.session_state:

        if st.button("Run Detection"):

            # ✅ RESET OLD STATES
            for key in ["bbox", "heatmap", "explanation"]:
                if key in st.session_state:
                    del st.session_state[key]

            img = st.session_state["img"]

            emb = get_image_embedding(img)
            similar = find_similar_case(emb, load_memory())

            results = model.predict(st.session_state["path"])
            r = results[0]

            st.session_state["detected"] = cv2.cvtColor(
                r.plot(),
                cv2.COLOR_BGR2RGB
            )

            # default values
            x1 = y1 = x2 = y2 = None

            # ---------------- MEMORY CASE ----------------
            if similar:

                pred = similar["corrected_class"]
                confidence = 0.95

                st.success("🧠 Using learned memory")

                x1, y1, x2, y2 = map(int, similar["bbox"])

                st.session_state["bbox"] = (x1, y1, x2, y2)

                img_copy = st.session_state["img_rgb"].copy()

                cv2.rectangle(
                    img_copy,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                st.session_state["detected"] = img_copy

                st.session_state["heatmap"] = generate_heatmap(
                    st.session_state["img_rgb"],
                    (x1, y1, x2, y2)
                )

            # ---------------- NORMAL YOLO DETECTION ----------------
            elif r.boxes is not None and len(r.boxes) > 0:

                pred = model.names[int(r.boxes[0].cls[0])]

                confidence = float(r.boxes[0].conf[0])

                x1, y1, x2, y2 = r.boxes[0].xyxy[0].tolist()

                st.session_state["bbox"] = (x1, y1, x2, y2)

                st.session_state["heatmap"] = generate_heatmap(
                    st.session_state["img_rgb"],
                    (x1, y1, x2, y2)
                )

            # ---------------- NO TUMOR ----------------
            else:

                pred = "No Tumor"

                confidence = 0.5
            if x1 is not None:

                img_h, img_w = st.session_state["img_rgb"].shape[:2]

                box_area = (x2 - x1) * (y2 - y1)
                image_area = img_w * img_h
                area_percent = (box_area / image_area) * 100

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Location logic
                loc_x = "Left" if cx < img_w/3 else "Center" if cx < 2*img_w/3 else "Right"
                loc_y = "Upper" if cy < img_h/3 else "Middle" if cy < 2*img_h/3 else "Lower"

                location = f"{loc_y}-{loc_x}"

            else:
                area_percent = 0
                location = "Not Found"    

            st.session_state["pred"] = pred
            st.session_state["explanation"] = {
                "Confidence (%)": round(confidence * 100, 2),
                "Location": location,
                "Area Covered (%)": round(area_percent, 2)
            }
            st.session_state["confidence"] = confidence


        if "detected" in st.session_state:
            st.image(st.session_state["detected"])

        if "heatmap" in st.session_state:
            st.image(st.session_state["heatmap"])

        if "pred" in st.session_state:
            st.write("Prediction:", st.session_state["pred"])
        if "pred" in st.session_state:

            st.subheader("🧠 Prediction Result")

            st.write(f"*Prediction:* {st.session_state['pred']}")
            st.write(f"*Confidence:* {round(st.session_state['confidence'] * 100, 2)}%")

    # =========================
    # AI REPORT GENERATION
    # =========================

    if st.button("📄 Generate AI Medical Report"):

        report = generate_report(
            st.session_state["pred"],
            st.session_state["confidence"] * 100,
            st.session_state["explanation"]["Location"],
            st.session_state["explanation"]["Area Covered (%)"]
        )

        st.subheader("📋 AI Generated Report")

        st.write(report)

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name="brain_tumor_report.txt",
            mime="text/plain"
        )

   # ---------------- EXPLANATION ----------------
st.subheader("🔥 Explainable AI")

if "pred" in st.session_state and st.session_state["pred"] != "No Tumor" and "bbox" in st.session_state:

    exp = st.session_state["explanation"]

    st.write(f"• Confidence (%): *{exp['Confidence (%)']}%*")
    st.write(f"• Location: *{exp['Location']}*")
    st.write(f"• Area Covered (%): *{exp['Area Covered (%)']}%*")

    # st.write("""
    # The model focuses on a specific region...
    # """)

# else:
#     st.write("""
#     No strong abnormal region was detected.
#     """)
metrics = format_metrics()

if metrics:
    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", round(metrics["Accuracy"], 3))
    col2.metric("Precision", round(metrics["Precision"], 3))
    col3.metric("Recall", round(metrics["Recall"], 3))
    col4.metric("F1 Score", round(metrics["F1"], 3))
                 

with tab3:
    if "pred" in st.session_state:

        st.subheader("Doctor Feedback")

        decision = st.selectbox("Decision", ["Accepted", "Corrected"])

        # Show corrected class only if needed
        if decision == "Corrected":
            corrected = st.selectbox("Corrected Class", CLASS_NAMES)
        else:
            corrected = st.session_state["pred"]

        # Manual annotation ONLY if corrected
        if decision == "Corrected":
            st.subheader("Draw Correct Bounding Box")

            img_pil = Image.open(st.session_state["path"])
            w, h = img_pil.size

            canvas = st_canvas(
                background_image=img_pil,
                height=h,
                width=w,
                drawing_mode="rect",
                key="canvas_feedback"
            )

            if canvas.json_data and len(canvas.json_data["objects"]) > 0:
                obj = canvas.json_data["objects"][-1]

                st.session_state["manual_bbox"] = (
                    obj["left"],
                    obj["top"],
                    obj["left"] + obj["width"],
                    obj["top"] + obj["height"]
                )

                st.success("✅ Bounding box captured")

        # ✅ SAVE BUTTON (ALL LOGIC INSIDE THIS)
        if st.button("Save Feedback"):

            # Require bbox if corrected
            if decision == "Corrected" and "manual_bbox" not in st.session_state:
                st.error("⚠️ Please draw bounding box before saving")
                st.stop()

            # Save CSV
            df = pd.read_csv(FEEDBACK_FILE) if os.path.exists(FEEDBACK_FILE) else pd.DataFrame()

            new_row = pd.DataFrame([{
                "decision": decision,
                "predicted": st.session_state["pred"],
                "corrected": corrected,
                "time": datetime.now()
            }])

            df = pd.concat([df, new_row])
            df.to_csv(FEEDBACK_FILE, index=False)

            # ✅ Save dataset + memory ONLY here
            if decision == "Corrected":
                save_feedback_dataset(st.session_state["path"], corrected)

                img = cv2.imread(st.session_state["path"])
                emb = get_image_embedding(img)

                memory = load_memory()
                memory.append({
                    "embedding": emb,
                    "corrected_class": corrected,
                    "bbox": st.session_state["manual_bbox"]
                })
                save_memory(memory)

            st.success("✅ Feedback Saved!")

# -------- History --------
with tab4:
    if os.path.exists(FEEDBACK_FILE):
        st.dataframe(pd.read_csv(FEEDBACK_FILE))

# -------- Improvement --------
with tab5:
    count = get_corrected_count()
    st.write("Corrected:", count)

    if count>=30:
        st.warning("30 reached → retrain?")
        if st.button("Yes"):
            retrain_model_fast()

    if st.button("Manual Retrain"):
        retrain_model_fast()