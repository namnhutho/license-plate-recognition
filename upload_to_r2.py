import boto3
from botocore.client import Config
from io import BytesIO
import cv2
from dotenv import load_dotenv
import os
load_dotenv()
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
PREFIX=os.getenv("PREFIX")
# === Khởi tạo client R2 (dùng chuẩn giao tiếp S3) ===
s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

def upload_image_to_r2(frame, filename):
    try:
        # Encode ảnh sang định dạng JPG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("⚠️ Không thể mã hóa ảnh thành JPEG")
            return None

        image_bytes = BytesIO(buffer.tobytes())
        key = f"{PREFIX}/{filename}"

        # Upload lên R2
        s3.upload_fileobj(
            image_bytes,
            R2_BUCKET,
            key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        # print(f"✅ Uploaded to R2: {key}")
        return None  # Hoặc return URL nếu public

    except RuntimeError as e:
        print(f"❌ RuntimeError khi upload: {e}")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn khi upload: {e}")

