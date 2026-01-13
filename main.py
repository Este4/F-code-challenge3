import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols





from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. LOAD DỮ LIỆU ---
print("1. Đang đọc dữ liệu...") 
df = pd.read_csv('data.csv')

# sap xep theo thoi gian de tinh toan
df = df.sort_values(by='Thang_Giao_Dich')

# Tạo thêm cột Giá tháng trước để làm biến đầu vào cho mô hình (Time Series)
df['Gia_Thang_Truoc'] = df['Gia_Nha_TyVND'].shift(1)

df['Gia_Thang_Truoc'] = df['Gia_Thang_Truoc'].fillna(0) # Điền giá trị thiếu tháng đầu
# Loại bỏ các dòng có giá trị thiếu (NaN)
df_model = df.dropna()

# Tách riêng phần "Sốt đất" ra để vẽ màu (nhưng vẫn để AI học để thấy nó sai)
# Trong thực tế, ta nên lọc bỏ Outlier khi train, nhưng ở đây ta giữ lại 
# để chứng minh luận điểm: "Mô hình sai khi thị trường điên loạn".

# --- 2. CHUẨN BỊ TRAIN ---
# Input: Thang, Khoảng cách, Giá tháng trước (Time Series)
X = df[['Thang_Giao_Dich', 'Khoang_Cach_KM', 'Gia_Thang_Truoc']]
y = df['Gia_Nha_TyVND']

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
print("2. Đang huấn luyện mô hình Hồi quy tuyến tính...")
model = LinearRegression()
model.fit(X, y)

# --- 4. DỰ BÁO VÀ ĐÁNH GIÁ ---
y_pred = model.predict(X) # Nhờ máy dự đoán lại toàn bộ
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("-" * 30)
print(f"KẾT QUẢ MÔ HÌNH:")
print(f"  - Độ chính xác (R2): {r2:.2f} (Càng gần 1 càng tốt)")
print(f"  - Sai số trung bình (MSE): {mse:.2f}")
print(f"  - Công thức: Giá = {model.intercept_:.2f} + {model.coef_[0]:.2f}*DT + {model.coef_[1]:.2f}*KC + {model.coef_[2]:.2f}*Gia_Cu")
print("-" * 30)

# --- 5. VẼ BIỂU ĐỒ (VISUALIZATION) ---
print("3. Đang vẽ biểu đồ báo cáo...")
plt.figure(figsize=(12, 6))

# Vẽ đường giá thực tế
plt.plot(df['Thang_Giao_Dich'], y, marker='o', label='Gia thuc te', color='blue', alpha=0.6)

# Vẽ đường dự báo của AI
plt.plot(df['Thang_Giao_Dich'], y_pred, label='AI Dự báo (Model)', color='red', linestyle='--', linewidth=2)

# Tô màu vùng sốt đất (Dựa vào cột Trang_Thai)
# Lấy các tháng bị sốt đất
# 
bubble_data = df_model[df_model['Ghi_Chu'] == 'Sot_Dat']
if not bubble_data.empty:
    plt.scatter(bubble_data['Thang_Giao_Dich'], bubble_data['Gia_Nha_TyVND'], 
                color='orange', s=100, zorder=5, label='Sốt Đất (Bất thường)')
    plt.axvspan(40, 45, color='yellow', alpha=0.3, label='Giai đoạn Sốt ảo')


plt.title('So sánh Giá nhà Thực tế vs AI Dự báo (Có phân tích Chuỗi thời gian)')
plt.xlabel('Thời gian (Tháng)')
plt.ylabel('Giá nhà (Tỷ VNĐ)')
plt.legend()
plt.grid(True)

# Lưu ảnh để bạn E làm báo cáo
plt.savefig('bieu_do_phan_tich.png')
print("✅ Đã vẽ xong! Ảnh được lưu là 'bieu_do_phan_tich.png'")
plt.show()