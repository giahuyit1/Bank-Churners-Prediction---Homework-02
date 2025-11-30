#  CREDIT CARD CHURN PREDICTION

**Môn học:** PROGRAMMING FOR DATA SCIENCE  
**Bài tập:** Homework 2 - Numpy for Data Science  
**Sinh viên thực hiện:** Nguyễn Gia Huy - 23120047

---

## 📑 Mục lục (Table of Contents)
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Phương pháp thực hiện](#phương-pháp-thực-hiện)
4. [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Kết quả](#kết-quả)
7. [Cấu trúc dự án](#cấu-trúc-dự-án)
8. [Khó khăn & giải pháp](#khó-khăn-giải-pháp)
9. [Tác giả](#tác-giả)

---

## 1. Giới thiệu <a name="giới-thiệu"></a>

### Mô tả bài toán
Dự án này tập trung giải quyết bài toán **[Xác định khả năng rời đi của khách hàng sử dụng dịch vụ thẻ tín dụng]**

### Động lực và Ứng dụng
[Việc dự đoán khách hàng rời bỏ giúp doanh nghiệp đưa ra các chiến lược giữ chân khách hàng kịp thời].

### Mục tiêu cụ thể
* Sử dụng thành thạo thư viện **NumPy** để xử lý dữ liệu dạng bảng mà không dùng Pandas.
* Thực hiện các phân tích thống kê và trực quan hóa dữ liệu để hiểu rõ về tập dữ liệu.
* Xây dựng mô hình học máy (ví dụ: Logistic Regression) để đưa ra dự đoán.

---

## 2. Dataset <a name="dataset"></a>

* **Tên bộ dữ liệu:** [Credit Card customers].
* **Nguồn dữ liệu:** [https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers].
* **Kích thước:** 10127 mẫu và 9 đặc trưng (và mã định danh).
* **Mô tả các đặc trưng chính (Features):**
    * `Feature Attrition_Flag`: Trạng thái của khách hàng : 'Existing Customers nghĩa là khách hàng đang sử dụng dịch vụ,'Attrited Customer' nghĩa là khách hàng đã hủy dịch vụ
    * `Feature Customer_Age`: Tuổi của khách hàng
    * `Target`: Dự đoán Trạng thái của khách hàng dựa trên các đặc trưng khác

---

## 3. Phương pháp thực hiện <a name="phương-pháp-thực-hiện"></a>

Theo yêu cầu của đồ án, quy trình xử lý dữ liệu tuân thủ nghiêm ngặt việc sử dụng **NumPy**.

### 3.1. Quy trình xử lý dữ liệu (Data Preprocessing)
1.  **Đọc dữ liệu:** Sử dụng `numpy.genfromtxt` hoặc `numpy.loadtxt`.
2.  **Làm sạch dữ liệu:**
    * Kiểm tra tính hợp lệ và xử lý giá trị ngoại lai (Outliers).
3.  **Chuẩn hóa (Normalization/Standardization):**
    * Áp dụng Z-score standardization để đưa dữ liệu về phân phối chuẩn ($\mu=0, \sigma=1$) cho các thuật toán dựa trên Gradient.

### 3.2. Thuật toán (Logistic Regression)

* **Mô hình:** Hồi quy Logistic (Logistic Regression) và train random forest.
* **Hàm Giả thuyết (Hypothesis Function):**
    $$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

* **Hàm Mất mát (Cost Function - Binary Cross-Entropy):**
    $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

* **Thực hiện:** Sử dụng model có sẵn trong scikit learn

## 4. Cài đặt & Thiết lập <a name="cài-đặt--thiết-lập"></a>

Để chạy dự án này, hãy đảm bảo bạn đã cài đặt Python và các thư viện cần thiết.

```bash
# Clone repository
git clone https://github.com/giahuyit1/Bank-Churners-Prediction---Homework-02.git
# Cài đặt các thư viện
pip install -r requirements.txt
```
## Nội dung requirements.txt:



numpy <br>
matplotlib<br>
scikit-learn<br>
jupyter<br>

## 5. Hướng dẫn sử dụng <a name="hướng-dẫn-sử-dụng"></a>
Dự án được chia thành các notebook theo trình tự sau:

notebooks/01_data_exploration.ipynb: Chạy file này để xem phân tích khám phá dữ liệu (EDA) và trực quan hóa.

notebooks/02_preprocessing.ipynb: Chạy file này để thực hiện các bước tiền xử lý, làm sạch và chuẩn hóa dữ liệu (xuất ra file processed).

notebooks/03_modeling.ipynb: Chạy file này để huấn luyện mô hình và đánh giá kết quả.

## 6. Kết quả <a name="kết-quả"></a>
Hiệu suất mô hình (Model Performance)

Kết quả đánh giá trên tập kiểm tra (Test set)

|Metric	 |   Score |
|:---|:---|
|Accuracy |	0.80 | 
|Precision| 0.83 |
|Recall	 |   0.96 | 
|F1-Score| 0.89  |

# 7. Cấu trúc project

project/
├── README.md # Mô tả tổng quan dự án
├── requirements.txt # Liệt kê các thư viện cần thiết
├── data/ # Thư mục chứa dữ liệu
│ ├── raw/ # Dữ liệu gốc
│ │ └── BankChurners.csv
│ └── processed/ # Dữ liệu đã xử lý
│ ├── X_numpy.csv
│ └── Y_numpy.csv
├── notebooks/ # Jupyter Notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_modeling.ipynb
└── src/ # Mã nguồn tái sử dụng
├── init.py
├── data_processing.py
├── visualization.py
└── models.py
# 8. Khó khăn & giải pháp
|Thách thức (Khi chỉ dùng NumPy) | Giải pháp |
|---|---|
|Vectorization: Đảm bảo KHÔNG dùng for loops cho operations trên arrays.| Sử dụng các hàm np.dot(), np.sum(), và Broadcasting hiệu quả.|
|Xử lý String: Khó khăn khi xử lý biến phân loại (string) cho mã hóa.|	Áp dụng từ điển (Dictionary) để mapping string sang số nguyên trước khi áp dụng One-Hot Encoding thủ công.|






# 9.Tác giả
Nguyễn Gia Huy <br>
MSSV: 23120047
