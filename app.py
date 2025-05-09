import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import uuid, json,datetime,pytz
import psycopg2, time, threading
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from upload_to_r2 import upload_image_to_r22
from fastapi import FastAPI, Response
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from api_main import router as api_router

# Kết nối PostgreSQL
app = FastAPI()
app.include_router(api_router)
load_dotenv()

def insert_plate_record(frame_id, stream_id, vehicle_id, vehicle_bbox, plate_bbox, plate_text, plate_conf, image_name):
    query = """
        INSERT INTO plates (
            frame_id, stream_id, vehicle_id, vehicle_bbox, plate_bbox, plate_text, plate_conf, image_name, capture_time
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    capture_time = datetime.now()
    print("Giờ Việt Nam:", capture_time)

    cursor.execute(query, (
        frame_id,
        stream_id,
        vehicle_id,
        json.dumps(vehicle_bbox),
        json.dumps(plate_bbox),
        plate_text,
        plate_conf,
        image_name,
        capture_time.isoformat()
    ))
    conn.commit()

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9]', '', text)
def normalize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [round(x1/width, 6), round(y1/height, 6), round(x2/width, 6), round(y2/height, 6)]

def process_plate(roi, model_plate, ocr):
    plate_result = model_plate(roi)[0]
    boxes = plate_result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None

    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) == 0:
            xx1, yy1, xx2, yy2 = map(int, boxes.xyxy[i])
            plate_crop = roi[yy1:yy2, xx1:xx2]
            h, w = plate_crop.shape[:2]
            scale = 180 / w
            plate_resized = cv2.resize(plate_crop, (180, int(h*scale)), interpolation=cv2.INTER_CUBIC)
            plate_sharp = cv2.detailEnhance(plate_resized, sigma_s=10, sigma_r=0.2)
            cv2.imshow("PLATE ROI", plate_sharp)
            cv2.waitKey(1)
            result_ocr = ocr.ocr(plate_sharp, cls=True)
            if result_ocr and isinstance(result_ocr[0], list):
                lines = result_ocr[0]
                if 1 <= len(lines) <= 2:
                    full_text = ''.join([line[1][0] for line in lines])
                    avg_confident = lines[0][1][1] if len(lines) == 1 else sum(line[1][1] for line in lines) / len(lines)
                    return [xx1, yy1, xx2, yy2], clean_text(full_text), avg_confident
    return None, None, None

def tracking_loop(cap, model, model_plate, ocr):
    vehicle_id_map = {}
    stream_uuid4=str(uuid.uuid4())
    frame_id, last_result = 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % 5 == 0:
            results = model.track(source=frame, conf=0.5, persist=True, imgsz=640, stream=False, verbose=False)
            result = results[0]
            last_result = result
        
            if result.boxes.id is not None:
                for obj_id, cls, conf, box in zip(result.boxes.id.int().numpy(),
                                                result.boxes.cls.int().numpy(),
                                                result.boxes.conf.numpy(),
                                                result.boxes.xyxy.numpy()):
                    if conf < 0.5:
                        continue
                    # Gán/mapping id xe <> UUID
                    if obj_id not in vehicle_id_map:
                        vehicle_id_map[obj_id] = str(uuid.uuid4())
                    vehicle_uuid = vehicle_id_map[obj_id]

                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[y1:y2, x1:x2]
                    bbox, plate_text, confident_plate = process_plate(roi, model_plate, ocr)
                   
                    if bbox and plate_text and confident_plate:
                        px1, py1, px2, py2 = map(int, bbox)
                        # Vẽ khung
                        vehicle_norm = normalize_bbox([x1, y1, x2, y2], frame.shape[1], frame.shape[0])
                        plate_norm = normalize_bbox([x1 + px1, y1 + py1, x1 + px2, y1 + py2], frame.shape[1], frame.shape[0])
                        # Tạo frame_id tùy theo nguồn ảnh (camera hay video)
                        frame_id_uuid = str(uuid.uuid1())
                        name_image = f"frame_{frame_id_uuid}_id_{vehicle_uuid}.jpg"
                        insert_plate_record(
                            frame_id=frame_id_uuid,
                            stream_id=stream_uuid4,
                            vehicle_id=vehicle_uuid,
                            vehicle_bbox=vehicle_norm,
                            plate_bbox=plate_norm,
                            plate_text=plate_text,
                            plate_conf=confident_plate,
                            image_name=name_image
                        )
                        try:
                            if hasattr(executor1, "_shutdown") and not executor1._shutdown:
                                executor1.submit(upload_image_to_r22, frame.copy(), name_image)
                            else:
                                print("Executor1 đã shutdown, không thể submit!")
                        except RuntimeError as e:
                            print(f"Lỗi submit thread: {e}")
                        # upload_image_to_r22(frame, name_image)
                        # executor1.submit(upload_image_to_r22, frame.copy(), name_image)
                        cv2.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 255, 0), 2, cv2.LINE_AA)

                        (tw, th), baseline = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x1+px1, y1+py1-th-10), (x1+px1+tw, y1+py1), (0, 255, 0), -1)
                        cv2.putText(frame, plate_text, (x1+px1, y1+py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            frame = result.plot()
        else:
            if last_result:
                frame = last_result.plot()
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    conn = psycopg2.connect(
        dbname=os.getenv("DBNAME"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT")
    )
    executor1 = ThreadPoolExecutor(max_workers=4)  # Tùy máy, 4–8 là ổn
    cursor = conn.cursor()
    model = YOLO("D:\\InternAI\\aaaaaaaa\\best_bikev8n.onnx")
    model_plate = YOLO("D:\\InternAI\\aaaaaaaa\\best_platev8n.onnx")
    # cap = cv2.VideoCapture("D:\\InternAI\\model_2304\\gggg.mp4")
    # cap = cv2.VideoCapture("D:\\yolo_test\\B1_ontap\\test.MOV")
    cap=cv2.VideoCapture("D:\\InternAI\\aaaaaaaa\\video_demo_2804v2.mp4")
    # cap=cv2.VideoCapture('http://10.1.2.164:4747/video')
    # cap=cv2.VideoCapture(0)
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        det_db_box_thresh=0.4,
        det_db_thresh=0.25,
        det_db_unclip_ratio=1.2,
        rec_algorithm='CRNN',
        show_log=False,
        det=False
    )
    
    fastapi_thread = threading.Thread(target=start_fastapi)
    fastapi_thread.start()
    # uvicorn.run(app, port=8000)
    tracking_thread = threading.Thread(target=tracking_loop, args=(cap, model, model_plate, ocr))
    tracking_thread.start()
    # tracking_thread.join()
 