import cv2
import numpy as np
import matplotlib.pyplot as plt


# Hàm giúp hiển thị ảnh
def show_image(title, image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def canny_edge_detection_steps(image_path):
    # Đọc ảnh màu
    image = cv2.imread(image_path)

    # 1. Chuyển ảnh từ màu sang ảnh xám (grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image("Bước 1: Ảnh xám", gray_image)

    # 2. Làm mờ ảnh bằng bộ lọc Gaussian
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    show_image("Bước 2: Làm mờ ảnh", blurred_image)

    # 3. Tính toán gradient của ảnh bằng bộ lọc Sobel
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Tính gradient theo hướng X
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Tính gradient theo hướng Y

    # Tính độ lớn gradient và hướng của gradient
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    show_image("Bước 3: Gradient Magnitude", gradient_magnitude)
    show_image("Bước 3: Gradient Angle", gradient_angle, cmap='jet')

    # 4. Áp dụng Non-Maximum Suppression (NMS)
    rows, cols = gradient_magnitude.shape
    nms_image = np.zeros_like(gradient_magnitude, dtype=np.uint8)

    # Di chuyển qua từng pixel và áp dụng Non-Maximum Suppression
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Xác định góc hướng gradient gần nhất (góc chia làm 4 nhóm)
            angle = gradient_angle[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbor1 = gradient_magnitude[i, j + 1]
                neighbor2 = gradient_magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5):
                neighbor1 = gradient_magnitude[i + 1, j - 1]
                neighbor2 = gradient_magnitude[i - 1, j + 1]
            elif (67.5 <= angle < 112.5):
                neighbor1 = gradient_magnitude[i + 1, j]
                neighbor2 = gradient_magnitude[i - 1, j]
            else:  # (112.5 <= angle < 157.5)
                neighbor1 = gradient_magnitude[i - 1, j - 1]
                neighbor2 = gradient_magnitude[i + 1, j + 1]

            # Nếu pixel hiện tại là lớn nhất trong các pixel theo hướng gradient, giữ lại nó
            if gradient_magnitude[i, j] >= neighbor1 and gradient_magnitude[i, j] >= neighbor2:
                nms_image[i, j] = gradient_magnitude[i, j]

    show_image("Bước 4: Non-Maximum Suppression", nms_image)

    # 5. Áp dụng ngưỡng kép (Double Thresholding) và theo dõi biên (Hysteresis)
    low_threshold = 50
    high_threshold = 150

    # Xác định biên mạnh và biên yếu sau ngưỡng kép
    strong_edges = (nms_image >= high_threshold)  # Biên mạnh
    weak_edges = ((nms_image >= low_threshold) & (nms_image < high_threshold))  # Biên yếu

    # Tạo ảnh đầu ra
    final_edges = np.zeros_like(nms_image, dtype=np.uint8)
    final_edges[strong_edges] = 255  # Biên mạnh được gán là 255
    final_edges[weak_edges] = 75  # Biên yếu được gán là 75 (tạm thời)

    # Hiển thị kết quả sau ngưỡng kép
    show_image("Bước 5: Sau ngưỡng kép", final_edges)

    # 6. Theo dõi biên (Edge Tracking by Hysteresis) - liên kết các biên yếu với biên mạnh
    rows, cols = final_edges.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if final_edges[i, j] == 75:  # Biên yếu
                # Kiểm tra nếu có biên mạnh ở các pixel xung quanh
                if np.any(final_edges[i - 1:i + 2, j - 1:j + 2] == 255):  # Kiểm tra vùng lân cận
                    final_edges[i, j] = 255  # Kết nối với biên mạnh

    show_image("Bước 6: Kết quả Canny Edge Detection", final_edges)


# Thực thi với ảnh đầu vào
image_path = 'my_image.jpg'
canny_edge_detection_steps(image_path)
