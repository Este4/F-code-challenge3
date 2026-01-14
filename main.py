# --- 1. KHAI BÁO THƯ VIỆN ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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


# Kiểm tra giá trị rỗng
print("\nKiểm tra dữ liệu rỗng:")
print(df.isnull().sum())



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


print("\n" + "="*30)
print("PHẦN 6: ĐÁNH GIÁ CHUYÊN SÂU & KIỂM ĐỊNH")
print("="*30)

# 1. Tính toán phần dư (Residuals = Thực tế - Dự báo)
residuals = y - y_pred

# --- KIỂM ĐỊNH 1: ĐA CỘNG TUYẾN (VIF) ---
# Kiểm tra xem các biến đầu vào có bị trùng lặp thông tin không
print("\n[1] Kiểm tra Đa cộng tuyến (VIF):")
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)
print("-> Nhận xét: Nếu VIF > 10 là bị đa cộng tuyến nặng (cần loại bỏ biến).")

# --- KIỂM ĐỊNH 2: TÍNH CHUẨN CỦA PHẦN DƯ (NORMALITY) ---
# Phần dư cần phải phân phối chuẩn (hình chuông) thì mô hình mới tin cậy
print("\n[2] Kiểm tra phân phối chuẩn của phần dư:")
shapiro_test = stats.shapiro(residuals)
print(f"  - Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    print("  -> Kết luận: Phần dư tuân theo phân phối chuẩn (Tốt).")
else:
    print("  -> Kết luận: Phần dư KHÔNG phân phối chuẩn (Mô hình chưa tối ưu hoặc dữ liệu nhiễu).")

# --- VẼ BIỂU ĐỒ KIỂM ĐỊNH (VISUALIZATION) ---
plt.figure(figsize=(14, 6))

# Biểu đồ A: Histogram của phần dư (Xem có hình chuông không)
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, color='green')
plt.title('Phân phối của sai số (Residuals)')
plt.xlabel('Giá trị sai số (Tỷ VND)')
plt.ylabel('Tần suất')

# Biểu đồ B: Q-Q Plot (So sánh với đường chuẩn lý thuyết)
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Kiểm tra tính chuẩn)')

plt.tight_layout()
plt.show()

# --- KIỂM ĐỊNH 3: PHƯƠNG SAI ĐỒNG NHẤT (HOMOSCEDASTICITY) ---
# Kiểm tra xem lỗi có to dần khi giá nhà tăng không
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel('Giá trị dự báo (Fitted Values)')
plt.ylabel('Phần dư (Residuals)')
plt.title('Biểu đồ Residuals vs. Fitted Values')
plt.show()