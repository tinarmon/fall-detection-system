import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self, model_path='pose_landmarker_full.task'):
        # 1. ตั้งค่า MediaPipe Tasks API ด้วย BaseOptions และชี้ไปที่ไฟล์โมเดล
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # 2. กำหนดเงื่อนไขการทำงาน (เลือกโหมด IMAGE เพราะเราจะป้อนภาพให้มันทีละเฟรม)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # สร้างตัวตรวจจับ
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # 3. กำหนด Index แบบตัวเลขตรงๆ เพราะคำสั่ง mp.solutions โดนลบไปแล้ว
        # 11=ไหล่ซ้าย, 12=ไหล่ขวา, 23=สะโพกซ้าย, 24=สะโพกขวา, 25=เข่าซ้าย, 26=เข่าขวา
        self.TARGET_LANDMARKS = [11, 12, 23, 24, 25, 26]
        
        # กำหนดเส้นเชื่อม
        self.CONNECTIONS = [
            (11, 12), (11, 23), (12, 24),
            (23, 24), (23, 25), (24, 26)
        ]

    def process_frame(self, frame):
        """รับภาพเข้ามา ค้นหาจุด และวาดเส้นโครงร่างแบบใหม่"""
        # แปลงสีให้ MediaPipe อ่านได้
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # แปลงเป็นออบเจกต์ Image ตามกฎของ Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # ประมวลผลหาพิกัด
        detection_result = self.detector.detect(mp_image)
        
        points_px = {}
        
        # ถ้าระบบเจอโครงร่างมนุษย์ (pose_landmarks คือลิสต์ของจุดทั้งหมด)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0] # ดึงข้อมูลคนที่ 1
            h, w, _ = frame.shape
            
            # ดึงพิกัดเฉพาะ 6 จุดเป้าหมาย
            for idx in self.TARGET_LANDMARKS:
                lm = landmarks[idx]
                if lm.visibility > 0.5: # กรองความแม่นยำ
                    px, py = int(lm.x * w), int(lm.y * h)
                    points_px[idx] = (px, py)
                    cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)

            # วาดเส้นเชื่อม
            for p1, p2 in self.CONNECTIONS:
                if p1 in points_px and p2 in points_px:
                    cv2.line(frame, points_px[p1], points_px[p2], (255, 200, 0), 3)
                    
        return frame, points_px