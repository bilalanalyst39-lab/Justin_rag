# 🚀 Deployment Guide for Justin RAG System

## 📋 Prerequisites

### Required API Keys
- **OpenAI API Key**: For embeddings and chat completion
- **AssemblyAI API Key**: For audio transcription
- **Pinecone API Key**: For vector database

### Environment Setup
```bash
# Create .env file
OPENAI_API_KEY=your_openai_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=rag-index
```

## 🐳 Docker Deployment (Recommended)

### 1. Quick Start with Docker Compose
```bash
# Clone and navigate to project
cd Asif

# Set environment variables
export OPENAI_API_KEY=your_key
export ASSEMBLYAI_API_KEY=your_key
export PINECONE_API_KEY=your_key

# Build and run
docker-compose up --build
```

### 2. Access Points
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Frontend Only)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit dashboard
4. Deploy

**Required Secrets in Streamlit Cloud:**
```
OPENAI_API_KEY
ASSEMBLYAI_API_KEY
PINECONE_API_KEY
BACKEND_URL (your_backend_url)
```

### Option 2: Railway/Render (Backend)
1. Connect GitHub repository
2. Set environment variables
3. Deploy with Dockerfile

**Dockerfile for Backend:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 3: AWS ECS/Google Cloud Run
```bash
# Build and push to container registry
docker build -t justin-rag-backend .
docker tag justin-rag-backend:latest your-registry/justin-rag-backend:latest
docker push your-registry/justin-rag-backend:latest

# Deploy to cloud service
```

## 🔧 Configuration Details

### Backend Configuration
- **Port**: 8000
- **CORS**: Enabled for all origins
- **Health Check**: `/health` endpoint
- **Persistent Storage**: `permanent_transcripts/`, `downloaded_audios/`

### Frontend Configuration
- **Port**: 8501
- **Backend URL**: Configurable via `BACKEND_URL` environment variable
- **Theme**: Custom dark/light theme

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key | Required |
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_INDEX_NAME` | Vector index name | `rag-index` |
| `BACKEND_URL` | Backend API URL | `http://127.0.0.1:8000` |

## 📊 Monitoring & Scaling

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Frontend availability
curl http://localhost:8501
```

### Scaling Considerations
- **Backend**: Scale horizontally behind load balancer
- **Pinecone**: Serverless scaling handled by Pinecone
- **File Storage**: Use S3/Google Cloud Storage for production

## 🔒 Security Best Practices

1. **Environment Variables**: Never commit API keys
2. **CORS**: Restrict origins in production
3. **HTTPS**: Use SSL certificates
4. **Rate Limiting**: Implement API rate limits
5. **Input Validation**: Sanitize all user inputs

## 🚨 Troubleshooting

### Common Issues
1. **Connection Refused**: Check if backend is running
2. **API Key Errors**: Verify environment variables
3. **Memory Issues**: Increase Docker memory limits
4. **Slow Performance**: Check Pinecone index size

### Logs
```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Individual services
docker logs justin_complete_asif_backend_1
```

## 📞 Support

For deployment issues:
1. Check logs above
2. Verify all environment variables
3. Ensure API keys have proper permissions
4. Check network connectivity

---

**Deployment Status**: Ready for production deployment
**Last Updated**: 2026-01-16
