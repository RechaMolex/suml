FROM python:3.11-slim

WORKDIR /app

# Skopiuj pliki wymagań
COPY requirements.txt .

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj kod aplikacji
COPY . .

# Utwórz katalog dla danych
RUN mkdir -p /app/data

# Ustaw port
EXPOSE 8501

# Komenda uruchomienia
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]