from ultralytics import YOLO
import multiprocessing

# yaml会自动下载
def main():
    model = YOLO("ultralytics/cfg/models/11/yolo11-CAF.yaml")  # build a new model from scratch
# model = YOLO("d:/Data/yolov8s.pt")  # load a pretrained model (recommended for training)

        # Train the model
    results = model.train(data="Data.yaml", batch=16, epochs=100, imgsz=640, workers=0)

if __name__=='__main__':
    main()