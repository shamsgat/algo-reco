# ==============================
# Makefile pour déployer Streamlit sur Cloud Run
# ==============================

# Variables
PROJECT_ID := smart-quasar-478510-r3
IMAGE_NAME := streamlit-acceptation
REGION := europe-west1
BUCKET := my-bucket
PORT := 8080
TAG := latest

# ==============================
# 1️⃣ Build Docker image
# ==============================
build:
	@echo "🔹 Build Docker image..."
	gcloud builds submit --tag gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG) streamlit/

# ==============================
# 2️⃣ Déployer sur Cloud Run
# ==============================
deploy:
	@echo "🚀 Déploiement sur Cloud Run..."
	gcloud run deploy $(IMAGE_NAME) \
		--image gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--timeout 300

# ==============================
# 3️⃣ Nettoyer anciennes images locales Docker
# ==============================
clean:
	@echo "🧹 Nettoyage des images Docker locales..."
	docker rmi gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG) || true

# ==============================
# 4️⃣ Déployer en une seule commande
# ==============================
all: build deploy
	@echo "✅ Streamlit déployé sur Cloud Run"

