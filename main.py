import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Chuẩn bị dữ liệu ---
# Mình tạo nhanh tập data mẫu dựa trên yêu cầu của bạn
raw_data = {
    'Quận': ['Quận 9', 'Quận Tân Bình', 'Quận 9', 'Quận Tân Phú', 'Quận 9',
             'Quận 7', 'Quận 2', 'Quận 11', 'Quận Thủ Đức', 'Huyện Bình Chánh',
             'Quận 9', 'Quận Tân Phú', 'Quận 9', 'Quận 7', 'Quận 2'] * 7,
    'Diện_tích': [69, 74.1, 46.5, 65, 70, 70, 56.6, 20, 89, 55, 68, 74, 50, 65, 72] * 7,
    'Số_phòng': [2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2] * 7,
    'Giá_triệu': [2650, 3970, 678, 2870, 3000, 3200, 3800, 1570, 3500, 1500, 2586, 3950, 1640, 2360, 2850] * 7
}

df = pd.DataFrame(raw_data)

# Thêm một chút nhiễu cho dữ liệu có vẻ "thật" hơn, không bị quá khớp
np.random.seed(42)
df['Giá_triệu'] += np.random.normal(0, 100, len(df))
df['Diện_tích'] += np.random.normal(0, 2, len(df))

# Check xem có dòng nào trống không rồi bỏ luôn cho sạch
df = df.dropna()
print(f"Dữ liệu sẵn sàng với {len(df)} dòng.")

# --- Trực quan hóa (EDA) ---
# Chỉ giữ lại những biểu đồ thực sự cần thiết để giải thích cho khách/sếp
plt.figure(figsize=(16, 5))

# 1. Xem phân phối giá nhà
plt.subplot(1, 3, 1)
sns.histplot(df['Giá_triệu'], kde=True, color='teal')
plt.title('Phân phối giá nhà')

# 2. Quan hệ giữa Diện tích và Giá (Cái này quan trọng nhất)
plt.subplot(1, 3, 2)
sns.regplot(data=df, x='Diện_tích', y='Giá_triệu', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Diện tích vs Giá (Có đường hồi quy)')

# 3. So sánh giá giữa các Quận
plt.subplot(1, 3, 3)
df.groupby('Quận')['Giá_triệu'].mean().sort_values().plot(kind='barh', color='salmon')
plt.title('Giá trung bình theo khu vực')

plt.tight_layout()
plt.show()

# --- Tiền xử lý để chạy model ---
# Chuyển cột Quận (chữ) sang số bằng One-hot encoding
# Lưu ý: drop_first=True để tránh bẫy đa cộng tuyến (Dummy Variable Trap)
df_final = pd.get_dummies(df, columns=['Quận'], drop_first=True)

X = df_final.drop('Giá_triệu', axis=1)
y = df_final['Giá_triệu']

# Chia tập test 20% để kiểm tra độ chính xác sau khi train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Huấn luyện mô hình ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Đánh giá kết quả ---
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("-" * 30)
print(f"Chỉ số R2: {r2:.4f} (Giải thích được {r2*100:.1f}% dữ liệu)")
print(f"Sai số MAE: {mae:.2f} triệu")
print(f"Sai số RMSE: {rmse:.2f} triệu")
print("-" * 30)

# --- Phân tích hệ số (Để biết cái nào ảnh hưởng nhất) ---
importance = pd.DataFrame({
    'Yếu tố': X.columns,
    'Mức độ ảnh hưởng': model.coef_
}).sort_values(by='Mức độ ảnh hưởng', ascending=False)

print("\nHệ số hồi quy (Coefficients):")
print(importance)

# --- Thử dự đoán thực tế ---
# Ví dụ: Một căn 75m2, 3 phòng ở Quận 9
test_case = pd.DataFrame(0, index=[0], columns=X.columns)
test_case['Diện_tích'] = 75
test_case['Số_phòng'] = 3
if 'Quận_Quận 9' in test_case.columns:
    test_case['Quận_Quận 9'] = 1

predicted_val = model.predict(test_case)[0]
print(f"\n=> Dự đoán nhà 75m2, 3 phòng tại Quận 9: {predicted_val:.2f} triệu")

# --- Kiểm tra giả định (Residuals) ---
# Biểu đồ này để chứng minh mô hình chạy ổn, không bị lệch lạc
residuals = y_test - preds
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Giá dự đoán')
plt.ylabel('Sai số (Residuals)')
plt.title('Residuals Plot')

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()