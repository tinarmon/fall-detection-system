import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

TIME_STEPS = 10

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    print("--- ระบบประเมินความแม่นยำของ AI (Evaluation Mode) ---")
    
    # 1. โหลด Test Data (ข้อมูลคนใหม่ที่ไม่เคยใช้เทรน)
    test_file = 'data/test_dataset.csv' # แก้ Path
    try:
        df = pd.read_csv(test_file)
        X_raw = df.drop('label', axis=1).values 
        y_raw = df['label'].values
        X_test, y_true = create_sequences(X_raw, y_raw, TIME_STEPS)
        print(f"โหลดข้อมูลทดสอบ '{test_file}' สำเร็จ! จำนวน: {len(X_test)} ลำดับเหตุการณ์")
    except Exception as e:
        print(f"ไม่พบไฟล์ {test_file} กรุณาสร้าง Test Dataset ก่อนครับ")
        return

    # 2. โหลดโมเดล
    try:
        model = tf.keras.models.load_model('assets/fall_model.keras') # แก้ Path
        print("โหลดสมอง AI 'fall_model.keras' สำเร็จ!\n")
    except:
        print("ไม่พบไฟล์โมเดล กรุณาเทรนโมเดลก่อนครับ")
        return

    # 3. ให้ AI ทำนายผล (Predict)
    print("กำลังให้ AI ทำนายผลข้อมูลใหม่...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.6).astype(int).flatten() # แปลงความน่าจะเป็นให้เป็น 0 หรือ 1

    # 4. สรุปผลความแม่นยำสำหรับใส่ "บทที่ 5"
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"🎯 ความแม่นยำรวม (Overall Accuracy): {accuracy * 100:.2f}%")
    print("="*50)
    print("\n📊 ตาราง Confusion Matrix:")
    print(f"ทายว่า ปกติ(0) และเป็น ปกติ(0) จริงๆ : {conf_matrix[0][0]} ครั้ง")
    print(f"ทายว่า ล้ม(1) แต่จริงๆคือ ปกติ(0)   : {conf_matrix[0][1]} ครั้ง (False Alarm)")
    print(f"ทายว่า ปกติ(0) แต่จริงๆคือ ล้ม(1)    : {conf_matrix[1][0]} ครั้ง (Missed)")
    print(f"ทายว่า ล้ม(1) และเป็น ล้ม(1) จริงๆ   : {conf_matrix[1][1]} ครั้ง")
    print("-" * 50)
    print("\n📈 รายงาน Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fall (1)']))

if __name__ == "__main__":
    main()