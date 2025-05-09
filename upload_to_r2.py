import boto3
from botocore.client import Config
from io import BytesIO
import cv2
import os
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

# Cấu hình thông tin kết nối R2
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
PREFIX = os.getenv("PREFIX", "uploads")  # prefix mặc định nếu không khai báo

# Khởi tạo client boto3 để kết nối với Cloudflare R2
s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

def upload_image_to_r22(frame, filename) -> str | None:
    """
    Upload 1 ảnh (dưới dạng numpy array) lên Cloudflare R2 với định dạng JPG.

    Args:
        frame (np.ndarray): Ảnh dạng OpenCV (BGR).
        filename (str): Tên file lưu trên R2 (VD: 'anh_xe.jpg').

    Returns:
        str | None: Trả về URL giả định (nếu cần), hoặc None nếu lỗi.
    """
    try:
        # Chuyển ảnh thành buffer JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("⚠️ Không thể mã hóa ảnh thành JPEG")
            return None

        # Tạo luồng byte từ ảnh
        image_bytes = BytesIO(buffer.tobytes())
        key = f"{PREFIX}/{filename}"
        print(f"✅ Upload thành công: {key}")
        # Upload lên R2
        s3.upload_fileobj(
            Fileobj=image_bytes,
            Bucket=R2_BUCKET,
            Key=key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        
        # Nếu bucket public, có thể trả về URL như sau:
        # return f"https://{R2_BUCKET}.{R2_ENDPOINT.replace('https://', '')}/{key}"
        return key

    except RuntimeError as e:
        print(f"❌ RuntimeError khi upload: {e}")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn khi upload: {e}")

    return None

# # =================== Cách sử dụng ===================

# # Ví dụ: Đọc ảnh từ file và upload lên R2
# image = cv2.imread("D:\\yolo_test\\thanglucvi.jpg")  # Đảm bảo file test.jpg tồn tại
# filename = "test10_upload.jpg"
# result = upload_image_to_r22(image, filename)

# if result:
#     print(f"Ảnh đã được upload thành công với key: {result}")
# else:
#     print("Có lỗi khi upload ảnh.")
