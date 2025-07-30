# 1. תמונת בסיס עם Python 3.10
FROM python:3.12-slim

# 2. הגדרת תיקיית עבודה
WORKDIR /app

# 3. העתק קובץ הדרישות והתקן תלויות
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. העתק את שאר קבצי הפרויקט
COPY . .

# 5. פתח פורט 8000 (או כל פורט שהשרת מאזין אליו)
EXPOSE 8000

# 6. פקודת הרצה (עדכן את שם הקובץ והשירות לפי הצורך)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
