
import pandas as pd

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("/content/drive/MyDrive/project_facebook_bigdata/omg/lable_data.csv")

# Kiểm tra dữ liệu
print(df.head())


import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("/content/drive/MyDrive/project_facebook_bigdata/omg/lable_data.csv")

# Kiểm tra và xử lý giá trị NaN trong trường "content"
df['content'].fillna('', inplace=True)  # Thay thế giá trị NaN bằng chuỗi trống

# Đọc danh sách từ dừng từ tệp văn bản
with open("/content/vietnamese-stopwords.txt", "r", encoding="utf-8") as file:
    custom_stopwords = set([line.strip() for line in file])

# Tiền xử lý văn bản tiếng Việt
def preprocess_text(text):
    # Chuyển văn bản thành chữ thường
    text = text.lower()
    # Tách từ
    text = ViTokenizer.tokenize(text)
    # Loại bỏ dấu câu và từ dừng
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return text

# Áp dụng tiền xử lý cho tất cả các bình luận
df['content'] = df['content'].apply(preprocess_text)

# Tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()

# Biểu diễn văn bản bằng TF-IDF
X = vectorizer.fit_transform(df['content'])

# Nhãn của các bình luận
y = df['label']


#chia: train luyện và test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train
from sklearn.svm import SVC

# Tạo mô hình SVM
model = SVC(kernel='linear')

# Huấn luyện mô hình
model.fit(X_train, y_train)

#đánh giá 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)

# Báo cáo phân loại
report = classification_report(y_test, y_pred)
print("Báo cáo phân loại:\n", report)

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
print("Ma trận nhầm lẫn:\n", conf_matrix)
