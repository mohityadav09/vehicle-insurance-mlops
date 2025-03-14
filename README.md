# 🚗 Vehicle Insurance Data Pipeline - MLOps Project  

## 📌 Overview  
This project demonstrates an end-to-end **MLOps pipeline** for managing **vehicle insurance data**. The goal is to create a scalable, production-ready machine learning pipeline covering **data ingestion, processing, model training, deployment, and CI/CD automation**.  

By following this guide, you'll understand how to set up a complete ML pipeline that integrates **MongoDB, AWS, Docker, and CI/CD** for automating model deployment.  

---

## 📁 Project Structure  
### 1️⃣ Setting Up the Project  
- **Initialize the project structure** by executing `template.py`, which generates the required folder hierarchy and placeholder files.  
- **Manage dependencies** using `setup.py` and `pyproject.toml`.  
- **Set up a virtual environment** and install dependencies:  
  ```bash
  conda create -n vehicle python=3.10 -y
  conda activate vehicle
  pip install -r requirements.txt
  pip list  # Verify installations
  ```

---

## 📊 Data Management with MongoDB  
### 2️⃣ Configuring MongoDB Atlas  
- **Create a MongoDB Atlas account** and set up a **free M0 cluster**.  
- Configure access credentials and obtain the **MongoDB connection string**.  

### 3️⃣ Storing Data in MongoDB  
- Add your dataset to a `notebook/` directory.  
- Create a Jupyter notebook (`mongoDB_demo.ipynb`) to push data into MongoDB.  
- Verify stored data in **MongoDB Atlas** via `Database > Browse Collections`.  

---

## 📝 Logging, Exception Handling, and EDA  
### 4️⃣ Implement Logging & Exception Handling  
- Create a structured logging and error-handling module.  
- Test it with a demo script (`demo.py`).  

### 5️⃣ Exploratory Data Analysis (EDA) & Feature Engineering  
- Perform **EDA** and **feature engineering** to prepare data for ML model training.  
- Save preprocessed data for later use in the pipeline.  

---

## 📥 Data Ingestion  
### 6️⃣ Building the Data Ingestion Pipeline  
- Define MongoDB connection logic in `configuration/mongo_db_connections.py`.  
- Implement ingestion modules in `data_access/` and `components/data_ingestion.py` to fetch and transform data.  
- Configure ingestion parameters in `entity/config_entity.py` and `entity/artifact_entity.py`.  
- Execute `demo.py` after setting the MongoDB connection as an environment variable:  
  ```bash
  export MONGODB_URL="mongodb+srv://<username>:<password>@cluster-url"
  ```

---

## 🔍 Data Validation, Transformation & Model Training  
### 7️⃣ Data Validation  
- Define a **data schema** in `config/schema.yaml`.  
- Implement **validation logic** in `utils/main_utils.py`.  

### 8️⃣ Data Transformation  
- Develop transformation steps in `components/data_transformation.py`.  
- Create an estimator module (`entity/estimator.py`) for feature engineering.  

### 9️⃣ Model Training  
- Implement **model training pipeline** in `components/model_trainer.py`.  
- Ensure integration with the **transformation pipeline**.  

---

## 🌐 AWS Setup for Model Evaluation & Deployment  
### 🔟 AWS Configuration  
- Log in to AWS and create an **IAM user** with `AdministratorAccess`.  
- Set AWS credentials as environment variables:  
  ```bash
  export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
  export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
  ```

### 1️⃣1️⃣ Storing Models on AWS S3  
- Create an S3 bucket (`my-model-mlopsproj`) in the `us-east-1` region.  
- Implement model storage/retrieval in `src/aws_storage.py` and `entity/s3_estimator.py`.  

---

## 🚀 Model Deployment & Prediction Pipeline  
### 1️⃣2️⃣ Model Evaluation & Pusher  
- Implement evaluation logic in `components/model_evaluation.py`.  
- Develop a **model pusher module** to update models in the pipeline.  

### 1️⃣3️⃣ API Integration & Web UI  
- Create a **prediction pipeline** in `app.py`.  
- Add static and template directories for a simple web UI.  

---

## 🔄 CI/CD Setup with Docker, GitHub Actions & AWS  
### 1️⃣4️⃣ Docker & GitHub Actions  
- Create a **Dockerfile** and `.dockerignore`.  
- Set up **GitHub Actions** for automated deployment.  
- Store AWS secrets in GitHub (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `ECR_REPO`).  



