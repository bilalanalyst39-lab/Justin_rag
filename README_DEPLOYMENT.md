# 🚀 Quick Deployment Guide

## ⚡ Fast Start (Docker)

```bash
# 1. Set environment variables
export OPENAI_API_KEY=your_openai_key
export ASSEMBLYAI_API_KEY=your_assemblyai_key  
export PINECONE_API_KEY=your_pinecone_key

# 2. Run with Docker Compose
docker-compose up --build

# 3. Access applications
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

## 🌐 Cloud Deployment

### Streamlit Cloud (Easiest)
1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub
4. Set secrets in app settings
5. Deploy!

### Railway/Render
1. Connect GitHub repository  
2. Use Dockerfile for backend
3. Set environment variables
4. Deploy automatically

## 🔑 Required Environment Variables

```
OPENAI_API_KEY=sk-...
ASSEMBLYAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-index
BACKEND_URL=http://localhost:8000
```

## 📱 Access Points After Deployment

- **Frontend**: Your Streamlit app URL
- **Backend API**: Your API URL + `/docs`
- **Health Check**: Your API URL + `/health`

## 🆘 Need Help?

Check `DEPLOYMENT.md` for detailed instructions.
