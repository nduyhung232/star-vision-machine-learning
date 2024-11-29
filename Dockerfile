# Sử dụng image Python chính thức
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file yêu cầu vào container
COPY requirements.txt .

# Cài đặt các thư viện
RUN pip install --upgrade pip && pip install -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Cấu hình lệnh chạy chương trình
CMD ["python", "app.py"]
