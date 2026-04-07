import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import csv 
import os  
from core.pose_estimator import PoseEstimator      # แก้ import
from core.angle_calculator import AngleCalculator  # แก้ import

def main():
    print("กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
    try:
        model = tf.keras.models.load_model('assets/fall_model.keras') # แก้ Path
    except Exception as e:
        print("กรุณาตรวจสอบว่ามีไฟล์ 'fall_model.keras' อยู่ในโฟลเดอร์นี้หรือไม่")
        return

    estimator = PoseEstimator('assets/pose_landmarker_full.task') # แก้ Path
    calculator = AngleCalculator()
    
    TIME_STEPS = 10
    sequence_buffer = deque(maxlen=TIME_STEPS)
    
    # -------------------------------------------------------------
    # [ส่วนที่เพิ่มใหม่] เตรียมไฟล์เก็บบันทึกข้อมูลจากระบบ Live
    os.makedirs('data', exist_ok=True)
    live_csv_file = 'data/live_collected_data.csv' # แก้ Path
    file_exists = os.path.isfile(live_csv_file)
    # เปิดไฟล์ทิ้งไว้เลย (Append mode)
    f_live = open(live_csv_file, mode='a', newline='')
    writer = csv.writer(f_live)
    
    if not file_exists:
        header = ['predicted_label', 'left_angle', 'right_angle']
        for target in [11, 12, 23, 24, 25, 26]:
            header.extend([f'x{target}', f'y{target}'])
        writer.writerow(header)
    # -------------------------------------------------------------

    cap = cv2.VideoCapture(0)
    window_name = 'Fall Detection System - LIVE'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 

    print("ระบบพร้อมทำงาน! กด 'q', 'ๆ' หรือ 'ESC' เพื่อออก")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, points_px, points_norm = estimator.process_frame(frame)
        
        is_valid_pose = False
        left_angle, right_angle = 0, 0

        if points_px:
            if all(k in points_px for k in [11, 12, 23, 24, 25, 26]):
                is_valid_pose = True
                left_angle = calculator.calculate_angle(points_px[11], points_px[23], points_px[25])
                right_angle = calculator.calculate_angle(points_px[12], points_px[24], points_px[26])

        if is_valid_pose:
            features = [left_angle, right_angle]
            for target in [11, 12, 23, 24, 25, 26]:
                features.extend([points_norm[target][0], points_norm[target][1]])
            
            sequence_buffer.append(features)
            
            if len(sequence_buffer) == TIME_STEPS:
                input_data = np.array(sequence_buffer).reshape(1, TIME_STEPS, len(features))
                prediction = model.predict(input_data, verbose=0)[0][0]
                
                # ตัดสินใจ Label: เตือนแดง=1, เขียว=0
                predicted_label = 1 if prediction > 0.6 else 0
                
                # -------------------------------------------------------------
                # [ส่วนที่เพิ่มใหม่] บันทึกข้อมูลของเฟรมปัจจุบันลง CSV
                row_data = [predicted_label] + features
                writer.writerow(row_data)
                # -------------------------------------------------------------
                
                if prediction > 0.6:  
                    cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 255), 20)
                    cv2.putText(processed_frame, "WARNING: FALL DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.putText(processed_frame, f"AI Confidence: {prediction*100:.1f}%", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(processed_frame, "Status: NORMAL", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(processed_frame, f"Risk: {prediction*100:.1f}%", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('ๆ') or (key & 0xFF == ord('q')) or key == 27:
            break

    # -------------------------------------------------------------
    # [ส่วนที่เพิ่มใหม่] ปิดไฟล์เมื่อเลิกใช้งาน
    f_live.close()
    # -------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()