# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Expose port for Streamlit
EXPOSE 8011

#run streamlit

CMD ["streamlit", "run", "app.py", "--server.port=8011", "--server.address=0.0.0.0"]