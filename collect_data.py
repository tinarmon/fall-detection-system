import cv2
import csv
import os
from core.pose_estimator import PoseEstimator       # แก้ import
from core.angle_calculator import AngleCalculator   # แก้ import

def main():
    estimator = PoseEstimator('assets/pose_landmarker_full.task') # แก้ Path
    calculator = AngleCalculator()
    
    cap = cv2.VideoCapture(0)
    
    window_name = 'Data Collection Mode'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 

    os.makedirs('data', exist_ok=True) # สร้างโฟลเดอร์ data เผื่อไว้
    csv_file = 'data/fall_dataset.csv' # แก้ Path
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            header = ['label', 'left_angle', 'right_angle']
            for target in [11, 12, 23, 24, 25, 26]:
                header.extend([f'x{target}', f'y{target}'])
            writer.writerow(header)

        print("\n--- โหมดเก็บข้อมูล (Data Collection) แบบใหม่: เปิด-ปิด ---")
        print("- กด 'n' 1 ครั้ง = เริ่มอัดท่าปกติ (Normal | Label: 0)")
        print("- กด 'f' 1 ครั้ง = เริ่มอัดท่าเสียการทรงตัว (Fall | Label: 1)")
        print("- กด 'p' หรือ 'Spacebar' = หยุดอัดชั่วคราว (Pause)")
        print("- กด 'q' หรือ 'ESC' = ออกจากโปรแกรม\n")

        # สร้างตัวแปรเก็บสถานะการทำงาน (เริ่มต้นที่โหมดหยุดพัก)
        current_mode = "PAUSED"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, points_px, points_norm = estimator.process_frame(frame)
            
            left_angle, right_angle = 0, 0
            is_valid_pose = False

            if points_px:
                if all(k in points_px for k in [11, 12, 23, 24, 25, 26]):
                    is_valid_pose = True
                    # เรายังใช้ points_px คำนวณมุมเหมือนเดิม เพื่อให้มุมสมจริงกับภาพหน้าจอ
                    left_angle = calculator.calculate_angle(points_px[11], points_px[23], points_px[25])
                    right_angle = calculator.calculate_angle(points_px[12], points_px[24], points_px[26])
                    
                    cv2.putText(processed_frame, f"L:{int(left_angle)}", (points_px[23][0]+20, points_px[23][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"R:{int(right_angle)}", (points_px[24][0]-80, points_px[24][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            key = cv2.waitKey(10)
            
            # 1. เปลี่ยนโหมดตามปุ่มที่กด (กดแค่ครั้งเดียว ไม่ต้องค้าง)
            if key == ord('n') or key == ord('ช'):
                current_mode = "NORMAL"
            elif key == ord('f') or key == ord('ด'):
                current_mode = "FALL"
            elif key == ord('p') or key == ord('ย') or key == 32: # 32 คือ Spacebar
                current_mode = "PAUSED"
            elif key == ord('q') or key == ord('ๆ') or (key & 0xFF == ord('q')) or key == 27:
                break

            # 2. บันทึกข้อมูลและแสดงผลตามสถานะ (current_mode)
            status_text = "PAUSED (Press N or F to Start)"
            color = (0, 0, 255) # สีแดง (หยุดพัก)

            if current_mode == "NORMAL":
                status_text = "Recording: NORMAL (0) ... [Press 'P' to Pause]"
                color = (0, 255, 0)
                if is_valid_pose:
                    row_data = [0, left_angle / 180.0, right_angle / 180.0]
                    for target in [11, 12, 23, 24, 25, 26]:
                        # ดึงพิกัด points_norm (0.0-1.0) ลงไฟล์ CSV แทน!
                        row_data.extend([points_norm[target][0], points_norm[target][1]])
                    writer.writerow(row_data)

            elif current_mode == "FALL":
                status_text = "Recording: FALL (1) ... [Press 'P' to Pause]"
                color = (0, 165, 255)
                if is_valid_pose:
                    row_data = [1, left_angle, right_angle]
                    for target in [11, 12, 23, 24, 25, 26]:
                        # ดึงพิกัด points_norm (0.0-1.0) ลงไฟล์ CSV แทน!
                        row_data.extend([points_norm[target][0], points_norm[target][1]])
                    writer.writerow(row_data)

            cv2.putText(processed_frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imshow(window_name, processed_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("บันทึกข้อมูลสำเร็จ! ตรวจสอบไฟล์ 'fall_dataset.csv'")

if __name__ == "__main__":
    main()