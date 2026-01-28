import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('ggplot') 

print("--- BƯỚC 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU ---")
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("File not found")
    exit()

df = df.rename(columns={
    'QUẬN HUYỆN': 'Quận', 'DIỆN TÍCH - M2': 'Diện_tích',
    'SỐ PHÒNG': 'Số_phòng', 'GIÁ - TRIỆU ĐỒNG': 'Giá_triệu'
})
df = df[['Quận', 'Diện_tích', 'Số_phòng', 'Giá_triệu']].dropna()

df = df[(df["Giá_triệu"] > 100) & (df["Diện_tích"] > 10)]

Q1 = df["Giá_triệu"].quantile(0.25)
Q3 = df["Giá_triệu"].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df["Giá_triệu"] >= (Q1 - 1.5 * IQR)) & (df["Giá_triệu"] <= (Q3 + 1.5 * IQR))].copy()

print(f"Số mẫu sau khi lọc: {len(df_clean)}")

print("\n--- BƯỚC 2: TẠO BIẾN TIME SERIES & ONE-HOT ---")
df_clean['Giá_kỳ_trước'] = df_clean['Giá_triệu'].shift(1)
df_clean = df_clean.dropna()

df_final = pd.get_dummies(df_clean, columns=['Quận'], drop_first=True)

X = df_final.drop('Giá_triệu', axis=1)
y = df_final['Giá_triệu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("\n--- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH OLS ---")
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")

print("\n--- BƯỚC 4: KIỂM TRA ĐA CỘNG TUYẾN (VIF) ---")
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False).head(5))

print("\n--- BƯỚC 5: TRỰC QUAN HÓA DỮ LIỆU ---")

plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:60], label='Giá Thực tế', color='#1f77b4', linewidth=2)
plt.plot(preds[:60], label='Giá Dự đoán', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('BIỂU ĐỒ 1: So sánh Thực tế vs Dự đoán (60 mẫu đầu)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(y_test - preds, kde=True, bins=30, color='green')
plt.title('BIỂU ĐỒ 2: Phân phối sai số (Residuals)', fontsize=14)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': preds})
results_df['Sai lệch Abs'] = np.abs(results_df['Thực tế'] - results_df['Dự đoán'])
top_errors = results_df.sort_values(by='Sai lệch Abs', ascending=False).head(5)

print("\n--- TOP 5 CA DỰ BÁO SAI NHẤT (LIMITATIONS) ---")
print(top_errors)

top_errors[['Thực tế', 'Dự đoán']].plot(kind='bar', figsize=(10, 6), color=['#2ca02c', '#d62728'])
plt.title('BIỂU ĐỒ 3: Phân tích giới hạn - Các ca biến động bất thường', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()