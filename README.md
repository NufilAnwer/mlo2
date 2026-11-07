
# VisionGuard — AI-Generated Image Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws&logoColor=white" alt="AWS">
</p>

<p align="center">
  <a href="https://github.com/NufilAnwer/mlo2/actions/workflows/ci.yml">
    <img src="https://github.com/NufilAnwer/mlo2/actions/workflows/ci.yml/badge.svg" alt="VisionGuard CI/CD">
  </a>
</p>

**VisionGuard** is an AI-powered image classification and MLOps project designed to distinguish between AI-generated and real images. It combines deep learning, cloud-based inference, Docker containerization, and a Flask web application to provide an end-to-end image detection solution.

---

## 📌 Project Information

| Field | Details |
|---|---|
| Project Name | VisionGuard |
| Project ID | 23L-0532_A2_Ops |
| Student | Nufil Anwer |
| Roll Number | 23L-0532 |
| Programming Language | Python 3.11 |
| Repository | [GitHub](https://github.com/NufilAnwer/mlo2) |

---

## 🚀 Key Features

- **AI Image Detection:** Classifies images as AI-generated or real.
- **Deep Learning:** Uses a trained TensorFlow/Keras model.
- **Cloud Inference:** Performs predictions through an AWS-hosted API.
- **Flask Integration:** Connects the web application with the remote inference service.
- **API Authentication:** Secures prediction requests using API key authentication.
- **Prediction History:** Stores and displays previous classification results.
- **Docker Containerization:** Packages the inference API into a Docker container.
- **AWS Deployment:** Hosts the inference service on an AWS EC2 instance.
- **CI/CD Automation:** Integrates GitHub Actions workflows.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.11 |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Image Processing | OpenCV |
| Experiment Tracking | MLflow |
| Data Version Control | DVC |
| Collaboration | DagsHub |
| Web Framework | Flask |
| Database | SQLite |
| Containerization | Docker |
| Cloud Platform | AWS EC2 |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

---

## 🏗️ System Architecture

VisionGuard uses a distributed architecture in which the Flask web application communicates with a separately deployed cloud inference API.

```text
                  USER
                    |
                    v
           Flask Web Application
                    |
                    v
             Backend Routes
                    |
                    | Image Upload
                    | API Key Authentication
                    v
             AWS EC2 Instance
                    |
                    v
             Inference API
                    |
                    v
          TensorFlow / Keras Model
                    |
                    v
            Image Classification
                    |
             +------+------+
             |             |
             v             v
        AI Generated      Real
             |             |
             +------+------+
                    |
                    v
           Prediction Response
                    |
                    v
           Flask Web Application
                    |
             +------+------+
             |             |
             v             v
         Result Page   Prediction History
```

### Prediction Workflow

1. The user uploads an image through the Flask interface.
2. The Flask backend forwards the image to the AWS inference API.
3. The API validates the API key and image request.
4. The TensorFlow/Keras model processes the image.
5. The API returns the predicted class and confidence information.
6. Flask displays the prediction result.
7. The prediction is stored in the application's database.

---

## 🧠 Model Information

| Parameter | Details |
|---|---|
| Framework | TensorFlow / Keras |
| Model File | `task_3_5_champion.keras` |
| Input Shape | `160 × 160 × 3` |
| Classification | Binary Image Classification |
| Decision Threshold | `0.4980708956718445` |
| Output Classes | AI Generated, Real |

The model generates a prediction score that is evaluated against the configured decision threshold to determine the final classification.

---

## ☁️ Cloud Inference API

The VisionGuard inference API is deployed on AWS EC2 and provides endpoints for model health monitoring and image classification.

**Base URL**

```text
http://3.26.146.78:8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks API and model health |
| `/predict` | POST | Classifies an uploaded image |

### Health Check

```http
GET /health
```

Example:

```text
http://3.26.146.78:8000/health
```

### Image Prediction

```http
POST /predict
```

**Request requirements:**

| Parameter | Type | Description |
|---|---|---|
| `image` | File | Image to classify |
| `X-API-Key` | Header | Required API authentication key |

**Example JSON response:**

```json
{
  "label": "AI Generated",
  "confidence": 0.9621,
  "ai_probability": 0.9621,
  "model_name": "task_3_5_champion",
  "threshold": 0.4980708956718445
}
```

*The response values are illustrative.*

### API Authentication

Prediction requests require an API key in the `X-API-Key` request header.

API credentials are stored in environment variables and must never be committed to the repository.

---

## 📂 Project Structure

```text
VisionGuard/
│
├── app/
│   ├── __init__.py
│   └── routes.py
│
├── models/
│   └── task_3_5_champion.keras
│
├── data/
├── src/
├── scripts/
├── outputs/
├── tests/
├── configs/
├── docs/
│
├── api_server.py
├── run.py
├── Dockerfile.api
├── requirements-api.txt
├── requirements-docker.txt
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NufilAnwer/mlo2.git
cd mlo2
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-docker.txt
```

### 4. Configure Environment Variables

Create a local `.env.api` file:

```env
VISIONGUARD_API_URL=http://3.26.146.78:8000
VISIONGUARD_API_KEY=your_api_key_here
```

Replace the placeholder with your actual API key.

**Important:** Never upload `.env.api`, private keys, or actual credentials to GitHub.

### 5. Run the Flask Application

```bash
python run.py
```

Open the application:

```text
http://127.0.0.1:8000
```

---

## 🐳 Docker Deployment

The inference API is packaged and deployed using Docker.

### Build Docker Image

```bash
docker build -f Dockerfile.api -t visionguard-api:task-7 .
```

### Run Docker Container

```bash
docker run -d \
  --name visionguard-api \
  -p 8000:8000 \
  --env-file .env.api \
  visionguard-api:task-7
```

### Verify Container

```bash
docker ps
```

### Test API Health

```bash
curl http://localhost:8000/health
```

---

## 🔄 CI/CD Pipeline

VisionGuard integrates GitHub Actions to automate project workflows.

Workflow configuration:

```text
.github/workflows/ci.yml
```

View workflow executions:

[![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/NufilAnwer/mlo2/actions)

The CI/CD workflow provides automated checks and supports the project's development and deployment process.

---

## 🔐 Security Considerations

- API key authentication protects prediction requests.
- Credentials are loaded through environment variables.
- Sensitive environment files are excluded from version control.
- Private SSH keys are not included in the repository.
- Production deployments should use HTTPS.
- Cloud security group rules should restrict access to trusted sources.

**Note:** The current assignment deployment uses HTTP. It is intended for demonstration and should not be considered production-secure.

---

## 📊 Project Status

- [x] Machine learning model integration
- [x] Cloud inference API implementation
- [x] API key authentication
- [x] Docker containerization
- [x] AWS EC2 deployment
- [x] Flask-to-cloud API integration
- [x] Image prediction functionality
- [x] Prediction history integration
- [x] Local database verification
- [x] GitHub repository integration
- [ ] HTTPS and production security hardening
- [ ] Advanced monitoring and model drift detection

---

## 🔮 Future Improvements

- HTTPS-enabled cloud API communication.
- Advanced inference monitoring and centralized logging.
- Model performance and data drift monitoring.
- Automated model retraining and version management.
- API rate limiting and enhanced access control.
- Expanded model evaluation and testing.

---

## 👨‍💻 Author

**Nufil Anwer**

BS Computer Science  
FAST National University of Computer and Emerging Sciences

- GitHub: [@NufilAnwer](https://github.com/NufilAnwer)
- Repository: [VisionGuard — mlo2](https://github.com/NufilAnwer/mlo2)

---

<p align="center">
  <b>VisionGuard — Bringing transparency to AI-generated imagery.</b>
</p>
