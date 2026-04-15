import cv2
import numpy as np
import config
from collections import deque
import tensorflow as tf
import csv
import os
import time
import tkinter as tk  # <--- โหลดไลบรารีสร้างหน้าต่าง GUI
from tkinter import simpledialog  # <--- โหลดไลบรารีสร้างหน้าต่าง Popup
from core.pose_estimator import PoseEstimator
from core.angle_calculator import AngleCalculator
from core.ui_manager import UIManager


def ask_tester_name(current_name="Subject_01"):
    """
    ฟังก์ชันสำหรับสร้างหน้าต่าง Popup ให้กรอกชื่อ โดยไม่ต้องใช้ Terminal
    """
    root = tk.Tk()
    root.withdraw()  # ซ่อนหน้าต่างหลักสีเทาๆ ทิ้งไป ให้เหลือแต่ Popup
    # เด้งหน้าต่างถามชื่อ
    name = simpledialog.askstring(
        "Tester Input", "กรุณากรอกชื่อหรือ ID ของผู้ทดสอบ:", initialvalue=current_name
    )
    root.destroy()  # เคลียร์หน่วยความจำหลังกรอกเสร็จ

    if name and name.strip():
        return name.strip()
    return "Unknown"


def main():
    print("=" * 50)
    print("🚶‍♂️ ระบบตรวจจับการเสียการทรงตัว (Live Prediction)")
    print("=" * 50)

    # 1. เรียกหน้าต่าง Popup ถามชื่อตั้งแต่เริ่มรันโปรแกรม
    tester_name = ask_tester_name("")
    print(f"\nยินดีต้อนรับคุณ {tester_name} ระบบกำลังเปิดกล้อง...\n")

    print("กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
    try:
        model = tf.keras.models.load_model("assets/fall_model.keras")
    except Exception as e:
        print("กรุณาตรวจสอบว่ามีไฟล์ 'fall_model.keras' อยู่ในโฟลเดอร์นี้หรือไม่")
        return

    estimator = PoseEstimator("assets/pose_landmarker_full.task")
    calculator = AngleCalculator()
    ui = UIManager()

    TIME_STEPS = 10
    sequence_buffer = deque(maxlen=TIME_STEPS)

    os.makedirs("data", exist_ok=True)
    live_csv_file = "data/live_collected_data.csv"
    file_exists = os.path.isfile(live_csv_file)

    f_live = open(live_csv_file, mode="a", newline="")
    writer = csv.writer(f_live)

    if not file_exists:
        header = ["tester_name", "predicted_label", "left_angle", "right_angle"]
        for target in [11, 12, 23, 24, 25, 26]:
            header.extend([f"x{target}", f"y{target}"])
        writer.writerow(header)

    print("กำลังเชื่อมต่อกล้อง...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)  # หน่วงเวลา 1 วินาทีให้ฮาร์ดแวร์กล้องตั้งตัว

    window_name = "Fall Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("ระบบพร้อมทำงาน! กด 'q' เพื่อออก หรือกด 'n' เพื่อเปลี่ยนชื่อผู้ทดสอบ")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        # 2. ดักจับภาพเสียบแบบเงียบๆ (ถ้าภาพไม่มา ให้ข้ามไปเฟรมถัดไปเลย ไม่ต้อง print รัวๆ)
        if not ret or frame is None or frame.shape[0] == 0:
            continue

        curr_time = time.time()
        # ป้องกัน Error หารด้วย 0 หากกล้องส่งภาพมาเร็วเกินไป
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        processed_frame, points_px, points_norm = estimator.process_frame(frame)

        if processed_frame is None or processed_frame.size == 0:
            processed_frame = frame.copy()  # fallback ใช้ frame ดิบแทน

        is_valid_pose = False
        left_angle, right_angle = 0, 0
        bbox = None

        if points_px:
            xs = [p[0] for p in points_px.values()]
            ys = [p[1] for p in points_px.values()]
            h, w, _ = processed_frame.shape

            min_x, max_x = max(0, min(xs) - 50), min(w, max(xs) + 50)
            min_y, max_y = max(0, min(ys) - 100), min(h, max(ys) + 50)
            bbox = (min_x, min_y, max_x, max_y)

            if all(k in points_px for k in [11, 12, 23, 24, 25, 26]):
                is_valid_pose = True
                left_angle = calculator.calculate_angle(
                    points_px[11], points_px[23], points_px[25]
                )
                right_angle = calculator.calculate_angle(
                    points_px[12], points_px[24], points_px[26]
                )

        prediction = 0.0
        status_text = "NORMAL"
        theme_color = (0, 255, 0)

        if is_valid_pose:
            features = [left_angle / 180.0, right_angle / 180.0]
            for target in [11, 12, 23, 24, 25, 26]:
                features.extend([points_norm[target][0], points_norm[target][1]])

            sequence_buffer.append(features)

            if len(sequence_buffer) == TIME_STEPS:
                input_data = np.array(sequence_buffer).reshape(
                    1, TIME_STEPS, len(features)
                )
                prediction = model.predict(input_data, verbose=0)[0][0]

                predicted_label = 1 if prediction > 0.6 else 0
                row_data = [tester_name, predicted_label] + features
                writer.writerow(row_data)

                if prediction > 0.6:
                    status_text = "FALL DETECTED"
                    theme_color = (0, 0, 255)

        processed_frame = ui.draw_hud(
            frame=processed_frame,
            tester_name=tester_name,
            fps=fps,
            status_text=status_text,
            prediction=prediction,
            theme_color=theme_color,
            bbox=bbox,
        )

        # 3. แสดงผลภาพปกติ (ไม่ต้องมี if/else เช็คให้ print รกจอแล้ว เพราะเราดักไว้ตั้งแต่ข้อ 2)
        if processed_frame is not None and processed_frame.size > 0:
            cv2.imshow(window_name, processed_frame)
        else:
            cv2.imshow(window_name, frame)  # fallback

        key = cv2.waitKey(1)

        if key == ord("q") or key == ord("ๆ") or (key & 0xFF == ord("q")) or key == 27:
            break
        elif key == ord("n") or key == ord("ช"):
            new_name = ask_tester_name(tester_name)
            if new_name:
                tester_name = new_name
                print(f"อัปเดตชื่อผู้ทดสอบเป็น: {tester_name}")

    f_live.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
