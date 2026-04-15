import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 1. ตั้งค่าไฮเปอร์พารามิเตอร์ (Hyperparameters)
TIME_STEPS = 10  # ให้ AI ดูพฤติกรรมย้อนหลัง 10 เฟรม เพื่อตัดสินใจว่าล้มหรือไม่
EPOCHS = 30  # จำนวนรอบที่ให้ AI อ่านหนังสือ (ชุดข้อมูล) ซ้ำๆ
BATCH_SIZE = 32  # จำนวนข้อมูลที่ป้อนให้ AI เรียนรู้ต่อ 1 ครั้ง


def load_and_preprocess_data(csv_file):
    print(f"กำลังโหลดข้อมูลจาก {csv_file}...")
    df = pd.read_csv(csv_file)

    # แยกฟีเจอร์ (X) และ ป้ายกำกับ (y)
    # ตัดคอลัมน์ 'label' ออกเพื่อเอาไปเป็นคำตอบ และเก็บส่วนที่เหลือเป็นฟีเจอร์
    X = df.drop("label", axis=1).values
    y = df["label"].values

    return X, y


def create_sequences(X, y, time_steps):
    """
    แปลงข้อมูลรายเฟรม ให้เป็นข้อมูลแบบลำดับเวลา (Sequence) สำหรับโมเดล GRU
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # มัดรวมข้อมูล 10 เฟรมติดกันเป็น 1 ก้อน
        Xs.append(X[i : (i + time_steps)])
        # ใช้ป้ายกำกับของเฟรมสุดท้ายในก้อนนั้นเป็นคำตอบ
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def main():
    # 2. เตรียมข้อมูล
    X_raw, y_raw = load_and_preprocess_data("data/fall_dataset.csv")  # แก้ Path
    print(f"จำนวนเฟรมข้อมูลทั้งหมด: {len(X_raw)} เฟรม")

    X_seq, y_seq = create_sequences(X_raw, y_raw, TIME_STEPS)
    print(f"จัดกลุ่มเป็น Sequence ละ {TIME_STEPS} เฟรม ได้ทั้งหมด: {len(X_seq)} ชุด")

    # แบ่งชุดข้อมูลสำหรับฝึกสอน (Train) 80% และทดสอบสอบ (Test) 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
    )

    # 3. สร้างสถาปัตยกรรมโมเดล GRU
    print("\n--- กำลังสร้างโมเดล GRU ---")
    model = Sequential(
        [
            # ชั้นที่ 1: GRU คอยจับความสัมพันธ์ของเวลา (รับข้อมูลขนาด TIME_STEPS x จำนวนฟีเจอร์)
            GRU(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2])),
            Dropout(0.2),  # ป้องกันโมเดลจำข้อสอบ (Overfitting)
            # ชั้นที่ 2: GRU เพื่อสกัดลักษณะเด่นที่ลึกขึ้น
            GRU(32),
            Dropout(0.2),
            # ชั้นที่ 3: Neural Network ธรรมดาช่วยตัดสินใจ
            Dense(16, activation="relu"),
            # ชั้นส่งออก: ใช้ Sigmoid เพื่อให้ค่าออกมาเป็น 0 (ปกติ) หรือ 1 (ล้ม)
            Dense(1, activation="sigmoid"),
        ]
    )

    # กำหนดวิธีการเรียนรู้และตัวชี้วัด
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()  # แสดงโครงสร้างสมอง AI

    # 4. เริ่มการฝึกสอน (Training)
    print("\n--- เริ่มฝึกสอน (Training) ---")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,  # แบ่งอีก 20% ของ Train ไว้เช็คความคืบหน้าระหว่างเทรน
    )

    # 5. ประเมินผลความแม่นยำ (Evaluation)
    print("\n--- ประเมินผลกับชุดข้อมูลทดสอบ (Test Set) ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"ความแม่นยำของ AI (Test Accuracy): {accuracy * 100:.2f}%\n")

    # 6. บันทึกสมอง AI เก็บไว้ใช้งาน
    os.makedirs("assets", exist_ok=True)
    model_filename = "assets/fall_model.keras"  # แก้ Path
    model.save(model_filename)
    print(f"บันทึกโมเดลสำเร็จ! ไฟล์สมอง AI ชื่อ: '{model_filename}'")


if __name__ == "__main__":
    main()
