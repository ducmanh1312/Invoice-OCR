
from fastapi import FastAPI, File, UploadFile
from pipe import img_process
from PIL import Image
from io import BytesIO
import cv2
import shutil
import uvicorn
import os




app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "File không phải là ảnh JPEG hoặc PNG"}
    
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(file_path)
    if img  is None:
        return {'error': 'Không thể mở ảnh với OpenCV'}
    res = img_process(img)
    return res


if __name__ == "__main__":
    uvicorn.run('api:app', host="127.0.0.1", port=8000, reload=True)