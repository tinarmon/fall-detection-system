import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from pose_estimator import PoseEstimator
from angle_calculator import AngleCalculator

def main():
    # 1. โหลดสมอง AI และโมดูลวิเคราะห์ภาพ
    print("กำลังโหลดโมเดล AI (ใช้เวลาสักครู่)...")
    try:
        model = tf.keras.models.load_model('fall_model.keras')
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        print("กรุณาตรวจสอบว่ามีไฟล์ 'fall_model.keras' อยู่ในโฟลเดอร์นี้หรือไม่")
        return

    estimator = PoseEstimator('pose_landmarker_full.task') 
    calculator = AngleCalculator()
    
    # 2. ตั้งค่าระบบความจำ (Buffer) สำหรับเก็บข้อมูลย้อนหลัง 10 เฟรม
    TIME_STEPS = 10
    sequence_buffer = deque(maxlen=TIME_STEPS)
    
    # เปิดกล้องและตั้งค่าหน้าต่าง
    cap = cv2.VideoCapture(0)
    window_name = 'Fall Detection System - LIVE'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 

    print("ระบบพร้อมทำงาน! กด 'q', 'ๆ' หรือ 'ESC' เพื่อออก")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, points_px = estimator.process_frame(frame)
        
        is_valid_pose = False
        left_angle, right_angle = 0, 0

        # ตรวจสอบพิกัดร่างกาย
        if points_px:
            if all(k in points_px for k in [11, 12, 23, 24, 25, 26]):
                is_valid_pose = True
                left_angle = calculator.calculate_angle(points_px[11], points_px[23], points_px[25])
                right_angle = calculator.calculate_angle(points_px[12], points_px[24], points_px[26])

        # 3. เตรียมข้อมูลและป้อนให้ AI ทำนายผล (Prediction)
        if is_valid_pose:
            # จัดเรียงข้อมูลให้ตรงกับไฟล์ CSV เป๊ะๆ (มุมซ้าย, มุมขวา, และพิกัด x,y ทั้ง 6 จุด รวม 14 ค่า)
            features = [left_angle, right_angle]
            for target in [11, 12, 23, 24, 25, 26]:
                features.extend([points_px[target][0], points_px[target][1]])
            
            # นำข้อมูลของเฟรมปัจจุบันใส่เข้าไปในหางคิว
            sequence_buffer.append(features)
            
            # ถ้าเก็บข้อมูลครบ 10 เฟรมแล้ว ให้ AI เริ่มทำงาน
            if len(sequence_buffer) == TIME_STEPS:
                # แปลงข้อมูลในคิวเป็น Numpy Array และปรับรูปทรงให้ตรงกับที่ GRU ต้องการ 
                # (1 ตัวอย่าง, 10 ช่วงเวลา, 14 ฟีเจอร์)
                input_data = np.array(sequence_buffer).reshape(1, TIME_STEPS, len(features))
                
                # AI ทำนายผล (ค่า prediction จะออกมาเป็นตัวเลข 0.00 ถึง 1.00)
                # เราใส่ verbose=0 เพื่อไม่ให้มันปริ้นต์ log รกเต็ม Terminal
                prediction = model.predict(input_data, verbose=0)[0][0]
                
                # 4. ระบบแจ้งเตือน (Alert System)
                # ตั้งเกณฑ์ (Threshold): ถ้าความน่าจะเป็นเกิน 60% (0.6) ถือว่าเกิดการล้ม
                if prediction > 0.6:  
                    # หน้าจอแจ้งเตือน: วาดกรอบสีแดงหนาๆ และขึ้นข้อความ WARNING
                    cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 255), 20)
                    cv2.putText(processed_frame, "WARNING: FALL DETECTED!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.putText(processed_frame, f"AI Confidence: {prediction*100:.1f}%", (50, 160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # หน้าจอสถานะปกติ: ขึ้นข้อความสีเขียว
                    cv2.putText(processed_frame, "Status: NORMAL", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    # โชว์เปอร์เซ็นต์ความเสี่ยงเล็กๆ ไว้ที่มุมขวาบน
                    cv2.putText(processed_frame, f"Risk: {prediction*100:.1f}%", (1000, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('ๆ') or (key & 0xFF == ord('q')) or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()