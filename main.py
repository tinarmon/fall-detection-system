import cv2
from pose_estimator import PoseEstimator
from angle_calculator import AngleCalculator

def main():
    estimator = PoseEstimator('pose_landmarker_full.task') 
    calculator = AngleCalculator()
    
    cap = cv2.VideoCapture(0)
    
    # ---------------------------------------------------------
    # ตั้งค่าหน้าต่างแบบ Normal ที่สามารถปรับขนาด/ขยายเต็มจอได้เอง
    window_name = 'Fall Detection System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # กำหนดขนาดเริ่มต้นให้ใหญ่พอดี (กว้าง 1280, สูง 720) 
    # สามารถกดปุ่ม Maximize ที่มุมขวาบนของหน้าต่างเพื่อขยายสุดจอได้
    cv2.resizeWindow(window_name, 1280, 720) 
    # ---------------------------------------------------------

    print("ระบบกำลังทำงาน... กด 'q', 'ๆ' หรือ 'ESC' เพื่อออก")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถรับภาพจากกล้องได้")
            break

        processed_frame, points_px = estimator.process_frame(frame)

        if points_px:
            if all(k in points_px for k in [11, 23, 25]):
                left_angle = calculator.calculate_angle(points_px[11], points_px[23], points_px[25])
                cv2.putText(processed_frame, f"{int(left_angle)}", 
                            (points_px[23][0] + 20, points_px[23][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if all(k in points_px for k in [12, 24, 26]):
                right_angle = calculator.calculate_angle(points_px[12], points_px[24], points_px[26])
                cv2.putText(processed_frame, f"{int(right_angle)}", 
                            (points_px[24][0] - 80, points_px[24][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        cv2.imshow(window_name, processed_frame)

        # ---------------------------------------------------------
        # รับค่าปุ่มคีย์บอร์ด 
        key = cv2.waitKey(10)
        
        # เช็คเงื่อนไข: กด 'q' (อังกฤษ) หรือ 'ๆ' (ไทย) หรือ 'ESC' (รหัส 27)
        if key == ord('q') or key == ord('ๆ') or (key & 0xFF == ord('q')) or key == 27:
            break
        # ---------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()