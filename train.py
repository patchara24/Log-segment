from ultralytics import YOLO
import torch
import multiprocessing as mp

def main():
    # โหลดโมเดล
    # โหลดโมเดลจากที่เทรนล่าสุด
    model = YOLO("runs/segment/train/weights/last.pt")

    # เทรน
    model.train(
        data="./data.yaml",
        epochs=50,
        imgsz=1024,     # ปรับขนาดภาพเป็น 1024
        batch=4,        # ลด batch size ลงเพราะภาพใหญ่กินเมมโมรี่
        device=0,     # ใช้ GPU
        workers=0    # ปลอดภัยสุดบน Windows
        
    )

    # validate
    model.val()
    
if __name__ == "__main__":
    mp.freeze_support()  # สำคัญบน Windows
    main()
