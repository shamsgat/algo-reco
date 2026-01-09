FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/streamlit

# Copier requirements depuis le même dossier
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier tout le code Streamlit
COPY . /app/streamlit/

EXPOSE 8080

# Lancer Streamlit sur le port fourni par Cloud Run
CMD sh -c "streamlit run /app/streamlit/app.py --server.port \$PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false"
