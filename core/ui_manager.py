import cv2

class UIManager:
    def __init__(self):
        # คุณสามารถตั้งค่าสีเริ่มต้นหรือฟอนต์ไว้ตรงนี้ได้
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hud(self, frame, tester_name, fps, status_text, prediction, theme_color, bbox):
        """
        ฟังก์ชันสำหรับวาด UI ทั้งหมดลงบนเฟรมวิดีโอ
        """
        h, w, _ = frame.shape

        # 1. วาด Bounding Box รอบตัวบุคคล (เส้นบาง 2px)
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), theme_color, 2)

        # 2. แถบ Dashboard กึ่งโปร่งใส มุมซ้ายบน
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (250, 160), (0, 0, 0), -1) 
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) # ปรับความโปร่งใส 60%

        # 3. ใส่ข้อความลงใน Dashboard
        cv2.putText(frame, f"TESTER : {tester_name}", (30, 45), self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS    : {int(fps)}", (30, 75), self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"STATUS : {status_text}", (30, 105), self.font, 0.7, theme_color, 2)

        # 4. Minimal Risk Bar (หลอดความเสี่ยง)
        bar_x, bar_y, bar_w, bar_h = 30, 130, 170, 10
        # พื้นหลังหลอดสีเทา
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        # ความยาวหลอดตามเปอร์เซ็นต์ prediction
        fill_w = int(bar_w * prediction)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), theme_color, -1)
        # ข้อความเปอร์เซ็นต์ท้ายหลอด
        cv2.putText(frame, f"{prediction*100:.0f}%", (bar_x + bar_w + 15, bar_y + 10), self.font, 0.5, (255, 255, 255), 1)

        # 5. วาดขอบจอสีแดงเตือนภัยเมื่อมีความเสี่ยงสูง
        if prediction > 0.6:
            cv2.rectangle(frame, (0, 0), (w, h), theme_color, 5)

        return frame