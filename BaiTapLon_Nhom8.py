import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model.conf = 0.5

def upload_image():
    # Chọn file ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Đọc ảnh và hiển thị ảnh gốc
    img = cv2.imread(file_path)
    display_image(img, original=True)

    # Phát hiện đối tượng
    results = model(img)
    thongtin = results.pandas().xyxy[0]

    # Vẽ khung đối tượng trên ảnh
    for index, row in thongtin.iterrows():
        label = row['name']
        confidence = row['confidence'] * 100  # Nhân 100 để có giá trị phần trăm
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Điều kiện để hiển thị nếu độ tin cậy cao hơn một ngưỡng nhất định
        if confidence > 50:  # Ngưỡng có thể điều chỉnh
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}%", (x_min, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị ảnh đã phát hiện đối tượng
    display_image(img, original=False)

    # Lưu kết quả vào file .txt
    save_results(thongtin)


def display_image(img, original):
    """ Hiển thị ảnh trong giao diện chính Tkinter """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((450, 450))  # Giới hạn kích thước ảnh để phù hợp với form
    img_tk = ImageTk.PhotoImage(img_pil)

    # Cập nhật hình ảnh vào label tương ứng
    if original:
        original_image_label.config(image=img_tk)
        original_image_label.image = img_tk  # Lưu tham chiếu để tránh bị giải phóng
    else:
        processed_image_label.config(image=img_tk)
        processed_image_label.image = img_tk  # Lưu tham chiếu để tránh bị giải phóng


def save_results(thongtin):
    output_file = "thongtin.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for index, row in thongtin.iterrows():
            label = row['name']
            confidence = row['confidence'] * 100  # Nhân 100 để có giá trị phần trăm
            x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            f.write(
                f"Đối tượng: {label}, Độ chính xác: {confidence:.2f}%, Tọa độ: ({x_min}, {y_min}, {x_max}, {y_max})\n")
    messagebox.showinfo("Thông báo", "Phát hiện đối tượng hoàn tất. Kết quả đã được lưu vào thongtin.txt")


# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Hệ thống phát hiện đối tượng và trích xuất đối tượng trong ảnh")
root.geometry("800x600")
root.configure(bg="#f0f0f0")  # Màu nền sáng

# Thêm logo và tên ở góc trái
logo_frame = tk.Frame(root, bg="#f0f0f0")
logo_frame.pack(side="top", anchor="w", padx=10, pady=10)

# Đường dẫn logo (Thay bằng đường dẫn tới logo của bạn)
logo_img = Image.open("C:/Users/tunga/PycharmProjects/pythonProject/yolov5/xulyanh/anh1.jpg")
logo_img.thumbnail((60, 60))  # Kích thước logo nhỏ gọn hơn
logo_tk = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(logo_frame, image=logo_tk, bg="#f0f0f0")
logo_label.pack(side="left")

# Tên người dùng
name_label = tk.Label(logo_frame, text="NHÓM 8", font=("Arial", 10, "bold"), bg="#f0f0f0", fg="#333")
name_label.pack(side="left", padx=5)  # Giảm khoảng cách tên

# Tiêu đề chính
title = tk.Label(root, text="Hệ thống phát hiện đối tượng và trích xuất đối tượng trong ảnh",
                 font=("Arial", 15, "bold"), bg="#f0f0f0", fg="#0000FF")
title.pack(pady=7)

# Nút chọn ảnh
upload_btn = tk.Button(root, text="Tải ảnh lên", command=upload_image, font=("Arial", 12), width=20, bg="#4CAF50",
                       fg="white")
upload_btn.pack(pady=10)

# Khung hiển thị ảnh gốc và ảnh đã phát hiện đối tượng
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)  # Thêm khoảng cách để khung ảnh dịch xuống dưới

# Khung chứa ảnh gốc
original_image_frame = tk.Frame(frame, bd=2, relief="ridge", bg="#ffffff", height=300)
original_image_frame.grid(row=0, column=0, padx=20, pady=30)
original_image_label_text = tk.Label(original_image_frame, text="Ảnh ban đầu", font=("Arial", 10, "bold"), fg="#333",
                                     bg="#ffffff")
original_image_label_text.pack()
original_image_label = tk.Label(original_image_frame, text="", font=("Arial", 10), fg="#333", bg="#ffffff")
original_image_label.pack(fill="both", expand=True)

# Khung chứa ảnh đã xử lý
processed_image_frame = tk.Frame(frame, bd=2, relief="ridge", bg="#ffffff", height=300)
processed_image_frame.grid(row=0, column=1, padx=20, pady=20)
processed_image_label_text = tk.Label(processed_image_frame, text="Ảnh đã xử lý", font=("Arial", 10, "bold"), fg="#333",
                                      bg="#ffffff")
processed_image_label_text.pack()
processed_image_label = tk.Label(processed_image_frame, text="", font=("Arial", 10), fg="#333", bg="#ffffff")
processed_image_label.pack(fill="both", expand=True)

# Khởi động ứng dụng
root.mainloop()
