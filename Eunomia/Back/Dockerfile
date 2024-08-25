FROM python:3.9.19-slim-bullseye

#Mise a jour de pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# définition du répertoire de travail du conteneur
WORKDIR /app

# copier les fichier du directory
COPY . .

# Installation des librairies
RUN pip install -r requirements.txt

#Exposition du port de strimlit
EXPOSE 8501 

HEALTHCHECK CMD curl --fail hhtp://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
