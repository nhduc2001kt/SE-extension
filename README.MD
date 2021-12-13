# **I. Isolation forest**


## **1. Các thư viện sử dụng**

Các thư viện sử dụng cho giải thuật:
*   **json:** Dùng để đọc file dữ liệu json được cung cấp.
*   **numpy:** Hỗ trợ các hàm xử lý mảng, ma trân,...
*   **pandas:** Hỗ trợ xử lý dữ liệu trên DataFrame
*   **plotly:** Hỗ trợ mô phỏng dữ liệu dưới dạng đồ thị
*   **sklearn:** Thư viện hỗ trợ một số thuật toán machine learning


## **2. Đọc dữ liệu**

Sử dụng hàm open để đọc file và hàm json.load để chuyển dữ liệu về dạng json, dữ liệu được đọc dưới dạng một ma trận n x 2.

Sử dụng **np.transpose** để chuyển dữ liệu về ma trận dạng 2 x n


## **3. Huấn luyện mô hình**

Sử dụng hàm IsolationForest để tạo một model hỗ trợ sẵn bên trong thư viện sklearn với các tham số truyền vào

*   n_estimators = 100: số lượng i tree tạo ra trong rừng iso 
*   max_samples = "auto": số lượng tối đa các cá thể lấy ra từ một lần lấy mẫu để tạo ra một cây i tree
*   contamination = 0.05: tỉ lệ anomaly
*   random_state = 42: chọn thuộc tính ngẫu nhiên trong tập thuộc tính và chọn threshold trên thuộc tính đã chọn trên để phân nhán trên i tree

## **4. Xử lý kết quả đầu ra**
Thêm một cột bất thường (**anomaly**) vào trong tập dữ liệu ban đầu.

## **5. Mô phòng kết quả (Visualization)**
Hiển thị kết quả thu được dưới dạng đồ thị.



# **II. STL decomposition**

## **1. Các thư viện sử dụng**

Các thư viện sử dụng cho giải thuật:
*   **json:** Dùng để đọc file dữ liệu json được cung cấp.
*   **numpy:** Hỗ trợ các hàm xử lý mảng, ma trân,...
*   **pandas:** Hỗ trợ xử lý dữ liệu trên DataFrame
*   **statsmodels:** Hỗ trợ các giải thuật khám phá dữ liệu, ước tính mô hình thống kê và thực hiện các kiểm tra thống kê
*   **datetime:** Thư viện thời gian của Python
*   **plotly:** Hỗ trợ mô phỏng dữ liệu dưới dạng đồ thị
*   **math:** Thư viện toán học của Python

## **2. Đọc dữ liệu**

Sử dụng hàm open để đọc file và hàm json.load để chuyển dữ liệu về dạng json, dữ liệu được đọc dưới dạng một ma trận n x 2.

Sử dụng **np.transpose** để chuyển dữ liệu về ma trận dạng 2 x n

Xuất dữ liệu dưới 1 data frame gồm 1 cột thời gian và một cột giá trị

## **3. Huấn luyện mô hình**

Sử dụng hàm **seasonal_decompose** để phân giải tập dữ liệu thành các thành phần:
*   Trend: xu hướng tăng giảm chung của đồ thị. 
*   Seasonal: xu hướng của thị theo từng chu kì thời gian
*   Residual : sai lệch ước lượng so với thực tế.

## **4. Xử lý kết quả từ Residual**

Chọn một ngưỡng threshold (cụ thể là 0.1).

Chỉ định những giá trị vượt ngưỡng là bất thường.

Thêm một cột bất thường (anomaly) vào trong tập dữ liệu ban đầu.

## **5. Mô phòng kết quả (Visualization)**
Mô phỏng kết quả thu được dưới dạng đồ thị.

## **6. Thay đổi giá trị ngưỡng (Threshold)**
Mô phỏng kết quả thu được dưới dạng đồ thị sau khi thay đổi ngưỡng.

# [Link google colab](https://colab.research.google.com/drive/1gLZewAIEwFwuMUr5s6NGLA3-YR4CooTM?usp=sharing)