import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import csv 
import os  
import time 
from core.pose_estimator import PoseEstimator
from core.angle_calculator import AngleCalculator
from core.ui_manager import UIManager  # <--- โหลดคลาส UI เข้ามาใหม่

def main():
    print("="*50)
    print("🚶‍♂️ ระบบตรวจจับการเสียการทรงตัว (Live Prediction)")
    print("="*50)
    tester_name = input("กรุณากรอกชื่อหรือ ID ของผู้ทดสอบ (เช่น Subject_01): ")
    if not tester_name.strip():
        tester_name = "Unknown"
    print(f"\nยินดีต้อนรับคุณ {tester_name} ระบบกำลังเริ่มทำงาน...\n")

    print("กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
    try:
        model = tf.keras.models.load_model('assets/fall_model.keras')
    except Exception as e:
        print("กรุณาตรวจสอบว่ามีไฟล์ 'fall_model.keras' อยู่ในโฟลเดอร์นี้หรือไม่")
        return

    # เรียกใช้งานคลาสต่างๆ
    estimator = PoseEstimator('assets/pose_landmarker_full.task') 
    calculator = AngleCalculator()
    ui = UIManager() # <--- ประกาศตัวแปรเรียกใช้ UI
    
    TIME_STEPS = 10
    sequence_buffer = deque(maxlen=TIME_STEPS)
    
    os.makedirs('data', exist_ok=True)
    live_csv_file = 'data/live_collected_data.csv'
    file_exists = os.path.isfile(live_csv_file)
    
    f_live = open(live_csv_file, mode='a', newline='')
    writer = csv.writer(f_live)
    
    if not file_exists:
        header = ['tester_name', 'predicted_label', 'left_angle', 'right_angle']
        for target in [11, 12, 23, 24, 25, 26]:
            header.extend([f'x{target}', f'y{target}'])
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    window_name = f'Fall Detection System - Tester: {tester_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 

    print("ระบบพร้อมทำงาน! กด 'q', 'ๆ' หรือ 'ESC' เพื่อออก")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        processed_frame, points_px, points_norm = estimator.process_frame(frame)
        
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
                left_angle = calculator.calculate_angle(points_px[11], points_px[23], points_px[25])
                right_angle = calculator.calculate_angle(points_px[12], points_px[24], points_px[26])

        prediction = 0.0
        status_text = "NORMAL"
        theme_color = (0, 255, 0) 

        if is_valid_pose:
            features = [left_angle, right_angle]
            for target in [11, 12, 23, 24, 25, 26]:
                features.extend([points_norm[target][0], points_norm[target][1]])
            
            sequence_buffer.append(features)
            
            if len(sequence_buffer) == TIME_STEPS:
                input_data = np.array(sequence_buffer).reshape(1, TIME_STEPS, len(features))
                prediction = model.predict(input_data, verbose=0)[0][0]
                
                predicted_label = 1 if prediction > 0.6 else 0
                row_data = [tester_name, predicted_label] + features
                writer.writerow(row_data)
                
                if prediction > 0.6:  
                    status_text = "FALL DETECTED"
                    theme_color = (0, 0, 255) 

        # ==========================================
        # โยนภาระวาดหน้าจอไปให้ UIManager จัดการบรรทัดเดียวจบ!
        # ==========================================
        processed_frame = ui.draw_hud(
            frame=processed_frame, 
            tester_name=tester_name, 
            fps=fps, 
            status_text=status_text, 
            prediction=prediction, 
            theme_color=theme_color, 
            bbox=bbox
        )

        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('ๆ') or (key & 0xFF == ord('q')) or key == 27:
            break

    f_live.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()