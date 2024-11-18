import cv2
import numpy as np
from skimage.feature import hog
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ฟังก์ชันสำหรับโหลดและสกัดฟีเจอร์จากภาพ
def extract_hog_features_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # ปรับขนาดภาพ

    # ใช้ HOG (Histogram of Oriented Gradients) ในการสกัดฟีเจอร์
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return hog_features

# โหลดโมเดลด้วย pickle
model_path = 'model/svm_Spiral.pkl'  # เปลี่ยนเป็นพาธโมเดลของคุณ
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# ใช้ Tkinter เพื่อให้ผู้ใช้เลือกไฟล์ภาพ
Tk().withdraw()  # ซ่อนหน้าต่างหลักของ Tkinter
image_path = askopenfilename(title='Select an Image', filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])

if image_path:
    # สกัดฟีเจอร์จากภาพ
    hog_features = extract_hog_features_from_image(image_path)
    hog_features = np.array(hog_features).reshape(1, -1)  # ปรับขนาดฟีเจอร์ให้เหมาะสมกับโมเดล

    # ทำนายผล
    prediction = loaded_model.predict(hog_features)

    # แสดงผลลัพธ์การทำนาย
    if prediction[0] == 0:
        print(f"The image '{image_path}' is predicted as: Healthy")
    else:
        print(f"The image '{image_path}' is predicted as: Parkinson")
else:
    print("No file selected.")
