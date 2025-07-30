# 1. תמונת בסיס עם Python 3.10
FROM python:3.12-slim

# 2. ספריית עבודה בתוך הקונטיינר
WORKDIR /app

# 3. העתק קובץ הדרישות והתקן את התלויות
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. העתק את שאר קבצי הפרויקט
COPY . .

# 5. פתח פורט 8000 (הפורט ש־uvicorn מאזין אליו)
EXPOSE 8000

# 6. פקודה להרצת השרת עם uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
