import os
import subprocess

def extract_frames(video_path, output_folder, fps):
    # Tạo thư mục lưu khung hình nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Câu lệnh FFmpeg để trích xuất khung hình
    command = [
        r"C:\Users\ADMIN\Downloads\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe",  # Đường dẫn đầy đủ đến FFmpeg
        "-i", video_path,           # Đường dẫn đến video đầu vào
        "-vf", f"fps={fps}",        # Số khung hình mỗi giây
        os.path.join(output_folder, "frame_%04d.png")  # Tên file đầu ra
    ]

    # Chạy lệnh
    try:
        subprocess.run(command, check=True)
        print(f"Khung hình đã được lưu trong thư mục: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi trích xuất khung hình: {e}")

# Đường dẫn video và thư mục lưu ảnh
video_file = "data/AWBubblyFlow.wmv"
output_dir = "data/frames"  # Thư mục lưu trữ khung hình
fps = 19               # Số khung hình muốn trích xuất mỗi giây

# Gọi hàm để tách khung hình
extract_frames(video_file, output_dir, fps)