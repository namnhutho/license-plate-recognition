from fastapi import FastAPI, APIRouter, Query
import psycopg2
import os, json
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import requests
import numpy as np
import base64
import cv2
import re
from fastapi.responses import JSONResponse
load_dotenv()
router=APIRouter()

R2_PUBLIC_URL=os.getenv("R2_PUBLIC_URL")
PREFIX=os.getenv("PREFIX")
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DBNAME"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT")
    )
conn = get_db_connection()

def is_valid_plate(plate_text):
    # Chuẩn hóa
    plate_text = plate_text.replace(' ', '').replace('.', '').upper()
    # Tổ hợp cho nhóm loạt: 1-2 số hoặc 1-2 chữ hoặc 1 chữ 1 số
    # 2 số đầu, sau đó nhóm từ 1 đến 2 ký tự (chữ hoặc số),
    # hoặc nhóm 1 chữ (có thể theo sau là 1 số)
    pattern = r'^\d{2}([A-Z]{1,2}|\d{1,2}|[A-Z]\d|\d[A-Z])-?\d{4,5}$'
    # Biển đặc biệt (ngoại giao, v.v.)
    special_pattern = r'^(NG|NN|QT|CV|CD|LD|KT)\d{2,3}-?\d{3,5}$'
    
    return (re.match(pattern, plate_text) is not None) or (re.match(special_pattern, plate_text) is not None)

@router.get("/test-db")
def test_database_connection():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        conn.close()
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class PlateRecord(BaseModel):
    frame_id: str
    vehicle_id: str
    vehicle_bbox: List[float]
    plate_bbox: List[float]
    plate_text: str
    image_name: str
    capture_time: datetime


# @app.get("/plates", response_model=List[PlateRecord])
# def get_all_plates():
   
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM plates")
#     rows = cursor.fetchall()
#     conn.close()
#     return [
#         PlateRecord(
#             frame_id=str(row[0]),
#             stream_id=str(row[1]),
#             vehicle_id=row[2],
#             vehicle_bbox=row[3],
#             plate_bbox=row[4],
#             plate_text=row[5],
#             image_name=row[6],
#             capture_time=row[7]
#         )
#         for row in rows
#     ]


@router.get("/plates_images")
def get_all_plates():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates")
    rows = cursor.fetchall()

    column_names = [
        'frame_id', 'stream_id', 'vehicle_id',
        'vehicle_bbox', 'plate_bbox', 'plate_text',
        'image_name', 'capture_time'
    ]

    def convert(row):
        return {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in zip(column_names, row)
        }

    result = [convert(row) for row in rows]
    print(result[0]['frame_id'])
    print(result[0]['image_name'])
    print(result[0]['plate_bbox'])
    image_name=result[0]['image_name']
    url = f"{R2_PUBLIC_URL}/{PREFIX}/{image_name}"
    response = requests.get(url)
    print(url , response)
    if response.status_code == 200:
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return result

def denormalize_bbox(normalized_bbox, width, height):
    x1, y1, x2, y2 = normalized_bbox
    return [
        round(x1 * width),  # Tọa độ x1 theo pixel
        round(y1 * height),  # Tọa độ y1 theo pixel
        round(x2 * width),  # Tọa độ x2 theo pixel
        round(y2 * height)  # Tọa độ y2 theo pixel
    ]
# Hàm để chuyển đổi ảnh sang Base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Chuyển ảnh thành định dạng JPEG
    return base64.b64encode(buffer).decode('utf-8')  # Mã hóa Base64 và trả về chuỗi

def process_plate_rows(rows):
    column_names = [
        'frame_id', 'stream_id', 'vehicle_id',
        'vehicle_bbox', 'plate_bbox', 'plate_text',
        'plate_conf', 'image_name', 'capture_time'
    ]

    def convert(row):
        return {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in zip(column_names, row)
        }

    result = [convert(row) for row in rows]
    filtered_result = []

    for item in result:
        frame_id = item["frame_id"]
        stream_id = item["stream_id"]
        vehicle_id = item["vehicle_id"]
        vehicle_box = item['vehicle_bbox']
        plate_box = item['plate_bbox']
        plate_text = item['plate_text']
        image_name = item['image_name']
        plate_conf = item['plate_conf']
        plate_check=is_valid_plate(str(plate_text))
        capture_time=item["capture_time"]
        
        url = f"{R2_PUBLIC_URL}/{PREFIX}/{image_name}"
        response = requests.get(url)
        if response.status_code == 200:
            image_data = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            height, width, _ = img.shape
            de_bike_bbox = denormalize_bbox(vehicle_box, width, height)
            de_plate_bbox = denormalize_bbox(plate_box, width, height)

            x1, y1, x2, y2 = de_bike_bbox
            px1, py1, px2, py2 = de_plate_bbox

            roi_img = img[y1:y2, x1:x2].copy()
            roi_px1, roi_py1 = px1 - x1, py1 - y1
            roi_px2, roi_py2 = px2 - x1, py2 - y1

            cv2.rectangle(roi_img, (roi_px1, roi_py1), (roi_px2, roi_py2), (0, 255, 0), 2)
            (tw, th), baseline = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(roi_img, (roi_px1, roi_py1 - th - 10), (roi_px1 + tw, roi_py1), (0, 255, 0), -1)
            text_to_draw = f"{plate_text} ({plate_conf * 100:.1f}%)"
            cv2.putText(roi_img, text_to_draw, (roi_px1, roi_py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            img_base64 = encode_image_to_base64(roi_img)
            filtered_result.append({
                "frame_id": frame_id,
                "stream_id": stream_id,
                "vehicle_id": vehicle_id,
                "plate_text": plate_text,
                "plate_check":plate_check,
                "plate_conf": plate_conf,
                "capture_time": capture_time,
                "image_url": url,
                # "image_base64": img_base64
            })
        else:
            print(f"Lỗi khi tải ảnh {image_name}: {response.status_code}")

    return filtered_result

@router.get("/plates")
def get_all_plates_new():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates")
    rows = cursor.fetchall()

    if not rows:
        return JSONResponse(content={"message": "Không tìm thấy dữ liệu"}, status_code=404)

    filtered_result=process_plate_rows(rows)
    if not filtered_result:
        return JSONResponse(content={"message": "Không có ảnh hợp lệ để hiển thị"}, status_code=404)

    filtered_result = [x for x in filtered_result if x.get('plate_check', False)]
    top_results = sorted(filtered_result, key=lambda x: x['plate_conf'], reverse=True)[:3]

    return JSONResponse(content=top_results)

@router.get("/vehicles/{vehicle_id}/plates")
def get_plates_by_vehicle(vehicle_id: str):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates WHERE vehicle_id = %s", (vehicle_id,))
    rows = cursor.fetchall()
    
    if not rows:
        return JSONResponse(content={"message": "Không tìm thấy dữ liệu"}, status_code=404)
    
    filtered_result=process_plate_rows(rows)
    if not filtered_result:
        return JSONResponse(content={"message": "Không có ảnh hợp lệ để hiển thị"}, status_code=404)
    # Lọc những kết quả có plate_check == True
    filtered_result = [x for x in filtered_result if x.get('plate_check', False)]
    top_results = sorted(filtered_result, key=lambda x: x['plate_conf'], reverse=True)[:3]
    return JSONResponse(content=top_results)

@router.get("/plates/by-time-range")
def get_plates_by_time_range(
    start_time: str = Query(..., description="Thời gian bắt đầu, định dạng YYYY-MM-DDTHH:MM:SS"),
    end_time: str = Query(..., description="Thời gian kết thúc, định dạng YYYY-MM-DDTHH:MM:SS")
):
    try:
        cursor = conn.cursor()
        query = """
            SELECT * FROM plates
            WHERE capture_time BETWEEN %s AND %s
            ORDER BY capture_time DESC
        """
        cursor.execute(query, (start_time, end_time))
        rows = cursor.fetchall()

        if not rows:
            return JSONResponse(content={"message": "Không tìm thấy dữ liệu trong khoảng thời gian"}, status_code=404)

        results = process_plate_rows(rows)
        if not results:
            return JSONResponse(content={"message": "Không có ảnh hợp lệ để hiển thị"}, status_code=404)

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/plates/search")
def search_plate_text(keyword: str = Query(..., description="Từ khóa cần tìm trong biển số")):
    try:
        cursor = conn.cursor()
        query = """
            SELECT * FROM plates
            WHERE plate_text ILIKE %s
            ORDER BY capture_time DESC
        """
        cursor.execute(query, (f"%{keyword}%",))
        rows = cursor.fetchall()

        if not rows:
            return JSONResponse(content={"message": "Không tìm thấy biển số khớp từ khóa"}, status_code=404)

        results = process_plate_rows(rows)
        if not results:
            return JSONResponse(content={"message": "Không có ảnh hợp lệ để hiển thị"}, status_code=404)

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/vehicles")
def get_all_vehicle_ids():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT vehicle_id FROM plates ORDER BY vehicle_id")
        rows = cursor.fetchall()

        if not rows:
            return JSONResponse(content={"message": "Không tìm thấy vehicle_id nào"}, status_code=404)

        vehicle_ids = [row[0] for row in rows]
        return JSONResponse(content={"vehicle_ids": vehicle_ids})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# if __name__ == "__main__":
#     uvicorn.run(router, port=8000)