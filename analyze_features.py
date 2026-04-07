import pandas as pd
import numpy as np

def main():
    os.makedirs('data', exist_ok=True)
    dataset_file = 'data/fall_dataset.csv'          # แก้ Path
    
    print(f"🔍 กำลังวิเคราะห์ข้อมูลทางสถิติจากไฟล์: '{dataset_file}'...")
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ {dataset_file} กรุณาตรวจสอบว่ามีไฟล์นี้อยู่ในโฟลเดอร์")
        return

    # 2. คำนวณค่า Mean และ Std โดยจัดกลุ่มตาม Label (0 และ 1)
    # ใช้ฟังก์ชัน groupby ของ pandas เพื่อหาค่าสถิติ
    stats_df = df.groupby('label').agg(['mean', 'std']).T

    # 3. จัดรูปแบบตารางใหม่ให้อ่านง่ายสำหรับนำไปใส่เล่มปริญญานิพนธ์
    # สร้าง DataFrame ใหม่เพื่อความสวยงาม
    summary = pd.DataFrame({
        'Normal_Mean (0)': df[df['label'] == 0].mean(),
        'Normal_Std (0)': df[df['label'] == 0].std(),
        'Fall_Mean (1)': df[df['label'] == 1].mean(),
        'Fall_Std (1)': df[df['label'] == 1].std()
    })
    
    # ตัดแถว 'label' ออกจากตารางสรุปเพราะไม่จำเป็นต้องใช้
    summary = summary.drop('label')
    
    # 4. คำนวณหา "ความแตกต่าง" เพื่อดูว่าตัวแปรไหนแยกแยะการล้มได้ดีที่สุด
    # เอาค่า Mean ท่าล้ม ลบด้วย Mean ท่าปกติ (ใส่ Absolute ให้เป็นค่าบวกเสมอ)
    summary['Mean_Difference'] = abs(summary['Normal_Mean (0)'] - summary['Fall_Mean (1)'])
    
    # เรียงลำดับจากตัวแปรที่ต่างกันมากที่สุด ไปหาน้อยที่สุด
    summary = summary.sort_values(by='Mean_Difference', ascending=False)

    print("\n" + "="*70)
    print("📊 สรุปค่าสถิติ Mean และ Std ของแต่ละตัวแปร (เรียงตามความแตกต่างสูงสุด)")
    print("="*70)
    # พิมพ์ผลลัพธ์ออกหน้าจอ (กำหนดทศนิยม 4 ตำแหน่ง)
    print(summary.round(4).to_string())
    print("="*70)

    # 5. บันทึกผลลัพธ์เป็นไฟล์ CSV เพื่อให้นำไปทำตารางใน Word ได้ง่ายๆ
    output_file = 'data/feature_statistics_report.csv' # แก้ Path
    summary.to_csv(output_file)
    print(f"\n✅ บันทึกตารางสรุปผลลงในไฟล์ '{output_file}' เรียบร้อยแล้ว!")
    print("💡 คำแนะนำ: คุณสามารถนำไฟล์นี้ไปเปิดใน Excel เพื่อคัดลอกลงเล่มบทที่ 4 หรือ 5 ได้ทันที")

if __name__ == "__main__":
    main()