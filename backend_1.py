import os
import shutil
import tempfile
import gc
import uuid
import requests
import feedparser
import hashlib
import json
import socket
from typing import List, Literal, Dict, Optional
from typing_extensions import TypedDict
from io import BytesIO

import assemblyai as aai
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from fpdf import FPDF
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client

# LangChain & LangGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from fastapi.middleware.cors import CORSMiddleware 
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import pinecone

# Set global socket timeout
socket.setdefaulttimeout(60)

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET_AUDIO = os.getenv("SUPABASE_BUCKET_AUDIO", "audio-files")
SUPABASE_BUCKET_TRANSCRIPTS = os.getenv("SUPABASE_BUCKET_TRANSCRIPTS", "transcripts")

# Temporary directories (for processing only, not permanent storage)
import tempfile
TEMP_TRANSCRIPT_DIR = os.path.join(tempfile.gettempdir(), "temp_transcripts")
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "temp_audios")
os.makedirs(TEMP_TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")

print(f"✅ AssemblyAI Key loaded: {bool(ASSEMBLYAI_API_KEY)}")
print(f"🔑 Key length: {len(ASSEMBLYAI_API_KEY) if ASSEMBLYAI_API_KEY else 0}")
print(f"✅ Supabase configured: {SUPABASE_URL}")

aai.settings.api_key = ASSEMBLYAI_API_KEY

# --- SUPABASE INITIALIZATION ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- SUPABASE HELPER FUNCTIONS ---
class SupabaseManager:
    def __init__(self):
        self.client = supabase
        self.audio_bucket = SUPABASE_BUCKET_AUDIO
        self.transcript_bucket = SUPABASE_BUCKET_TRANSCRIPTS
        self._ensure_buckets_exist()
    
    def _ensure_buckets_exist(self):
        """Ensure storage buckets exist"""
        try:
            # List existing buckets
            buckets = self.client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            
            # Create audio bucket if not exists
            if self.audio_bucket not in bucket_names:
                self.client.storage.create_bucket(
                    self.audio_bucket,
                    options={"public": False}
                )
                print(f"✅ Created Supabase bucket: {self.audio_bucket}")
            
            # Create transcript bucket if not exists
            if self.transcript_bucket not in bucket_names:
                self.client.storage.create_bucket(
                    self.transcript_bucket,
                    options={"public": False}
                )
                print(f"✅ Created Supabase bucket: {self.transcript_bucket}")
                
        except Exception as e:
            print(f"⚠️ Bucket creation info: {e}")
    
    def upload_audio(self, file_path: str, filename: str) -> Dict:
        """Upload audio file to Supabase storage"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to Supabase
            result = self.client.storage.from_(self.audio_bucket).upload(
                f"audios/{filename}",
                file_data,
                {"content-type": "audio/mpeg"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.audio_bucket).get_public_url(f"audios/{filename}")
            
            print(f"✅ Uploaded audio to Supabase: {filename}")
            return {
                "success": True,
                "url": public_url,
                "path": f"audios/{filename}",
                "bucket": self.audio_bucket
            }
        except Exception as e:
            print(f"❌ Error uploading audio to Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_pdf(self, file_path: str, filename: str) -> Dict:
        """Upload PDF file to Supabase storage"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to Supabase
            result = self.client.storage.from_(self.transcript_bucket).upload(
                f"pdfs/{filename}",
                file_data,
                {"content-type": "application/pdf"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.transcript_bucket).get_public_url(f"pdfs/{filename}")
            
            print(f"✅ Uploaded PDF to Supabase: {filename}")
            return {
                "success": True,
                "url": public_url,
                "path": f"pdfs/{filename}",
                "bucket": self.transcript_bucket
            }
        except Exception as e:
            print(f"❌ Error uploading PDF to Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> Dict:
        """Upload PDF from bytes to Supabase storage"""
        try:
            # Upload to Supabase
            result = self.client.storage.from_(self.transcript_bucket).upload(
                f"pdfs/{filename}",
                pdf_bytes,
                {"content-type": "application/pdf"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.transcript_bucket).get_public_url(f"pdfs/{filename}")
            
            print(f"✅ Uploaded PDF to Supabase: {filename}")
            return {
                "success": True,
                "url": public_url,
                "path": f"pdfs/{filename}",
                "bucket": self.transcript_bucket
            }
        except Exception as e:
            print(f"❌ Error uploading PDF to Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_audio(self, filename: str) -> bool:
        """Delete audio file from Supabase"""
        try:
            self.client.storage.from_(self.audio_bucket).remove([f"audios/{filename}"])
            print(f"🗑️ Deleted audio from Supabase: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error deleting audio: {e}")
            return False
    
    def delete_pdf(self, filename: str) -> bool:
        """Delete PDF file from Supabase"""
        try:
            self.client.storage.from_(self.transcript_bucket).remove([f"pdfs/{filename}"])
            print(f"🗑️ Deleted PDF from Supabase: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error deleting PDF: {e}")
            return False
    
    def get_audio_url(self, filename: str) -> str:
        """Get public URL for audio file"""
        return self.client.storage.from_(self.audio_bucket).get_public_url(f"audios/{filename}")
    
    def get_pdf_url(self, filename: str) -> str:
        """Get public URL for PDF file"""
        return self.client.storage.from_(self.transcript_bucket).get_public_url(f"pdfs/{filename}")

# Initialize Supabase Manager
supabase_manager = SupabaseManager()

# --- NETWORK VALIDATION ---
def validate_network_connectivity():
    """Test network connectivity before processing"""
    try:
        test_response = requests.get("https://api.assemblyai.com", timeout=10)
        print("✅ Network connectivity to AssemblyAI: OK")
        return True
    except Exception as e:
        print(f"⚠️ Network connectivity warning: {e}")
        return False

# Validate on startup
validate_network_connectivity()

# --- PINECONE INITIALIZATION ---
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings()

def get_pinecone_index():
    """Initialize or get Pinecone index"""
    try:
        if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
        return pinecone_client.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None


app = FastAPI(title="Professional Dual-Agent RAG with Speaker Diarization & Supabase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit local
        "http://localhost:8502",  # Streamlit alternative port
        "*"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS & STATE ---
class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class UrlRequest(BaseModel):
    url: str

class MultiUrlRequest(BaseModel):
    urls: List[str]

class ContentGenerationRequest(BaseModel):
    content: str
    content_type: str  # "cv", "transcript", "document"
    generation_type: str  # "interview_questions", "episode_brief", "summary"

class AgentState(TypedDict):
    messages: List[BaseMessage]
    question: str
    answer: str
    context: str
    sources: Optional[List[Dict]]
    episode_titles: Optional[List[str]]

# --- METADATA MANAGER FOR DEDUPLICATION ---
METADATA_FILE = "metadata.json"

class MetadataManager:
    def __init__(self):
        self.metadata_file = METADATA_FILE
        self.data = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load metadata from file"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        
        return {
            "files": {},
            "rss_feeds": {},
            "processed_episodes": {}
        }
    
    def save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def is_file_duplicate(self, filename: str, file_hash: str) -> bool:
        """Check if file already exists"""
        files = self.data["files"]
        for stored_filename, info in files.items():
            if info["hash"] == file_hash:
                return True
        return False
    
    def add_file(self, filename: str, file_hash: str, chunks_count: int, supabase_url: str = None):
        """Add file to metadata"""
        self.data["files"][filename] = {
            "hash": file_hash,
            "upload_time": str(uuid.uuid4()),
            "chunks_count": chunks_count,
            "supabase_url": supabase_url
        }
        self.save_metadata()
    
    def is_rss_feed_processed(self, feed_url: str) -> bool:
        """Check if RSS feed was processed before"""
        return feed_url in self.data["rss_feeds"]
    
    def get_processed_episodes(self, feed_url: str) -> set:
        """Get set of processed episode GUIDs for a feed"""
        processed_episodes = self.data["processed_episodes"]
        return {
            guid for guid, info in processed_episodes.items() 
            if info.get("feed_url") == feed_url
        }
    
    def add_rss_feed(self, feed_url: str, title: str):
        """Add RSS feed to metadata"""
        self.data["rss_feeds"][feed_url] = {
            "title": title,
            "last_fetched": str(uuid.uuid4()),
            "processed_episodes": 0
        }
        self.save_metadata()
    
    def add_episode(self, feed_url: str, episode_guid: str, title: str, audio_url: str = None, pdf_url: str = None):
        """Add episode to processed episodes"""
        self.data["processed_episodes"][episode_guid] = {
            "feed_url": feed_url,
            "title": title,
            "processed_time": str(uuid.uuid4()),
            "audio_url": audio_url,
            "pdf_url": pdf_url
        }
        
        if feed_url in self.data["rss_feeds"]:
            self.data["rss_feeds"][feed_url]["processed_episodes"] += 1
        
        self.save_metadata()

metadata_manager = MetadataManager()

# --- ENHANCED RSS PROCESSOR ---
class EnhancedRSSProcessor:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def parse_rss_feed(self, feed_url: str) -> Dict:
        """Parse RSS feed and return metadata + episodes"""
        try:
            response = requests.get(feed_url, headers=self.headers, timeout=15)
            feed = feedparser.parse(response.content)
            
            feed_info = {
                'title': getattr(feed.feed, 'title', ''),
                'description': getattr(feed.feed, 'description', ''),
                'total_episodes': len(feed.entries)
            }
            
            episodes = []
            for entry in feed.entries:
                episode = {
                    'guid': getattr(entry, 'id', entry.get('link', '')),
                    'title': getattr(entry, 'title', ''),
                    'description': getattr(entry, 'description', ''),
                    'audio_url': '',
                    'published_at': None
                }
                
                if hasattr(entry, 'enclosures') and entry.enclosures:
                    for enclosure in entry.enclosures:
                        if enclosure.type.startswith('audio/'):
                            episode['audio_url'] = enclosure.href
                            break
                
                if not episode['audio_url']:
                    episode['audio_url'] = getattr(entry, 'link', '')
                
                episodes.append(episode)
            
            return {
                'feed_info': feed_info,
                'episodes': episodes,
                'success': True
            }
            
        except Exception as e:
            return {
                'feed_info': {},
                'episodes': [],
                'success': False,
                'error': str(e)
            }
    
    def get_new_episodes(self, feed_url: str) -> Dict:
        """Get only new episodes for a feed with deduplication"""
        rss_data = self.parse_rss_feed(feed_url)
        if not rss_data['success']:
            return {
                "new_episodes": [],
                "is_new_feed": False,
                "feed_info": rss_data['feed_info'],
                "error": rss_data.get('error', 'Failed to parse RSS')
            }
        
        is_new_feed = not metadata_manager.is_rss_feed_processed(feed_url)
        
        if is_new_feed:
            return {
                "new_episodes": rss_data['episodes'],
                "is_new_feed": True,
                "feed_info": rss_data['feed_info'],
                "total_episodes": len(rss_data['episodes'])
            }
        
        processed_guids = metadata_manager.get_processed_episodes(feed_url)
        new_episodes = [
            ep for ep in rss_data['episodes'] 
            if ep['guid'] not in processed_guids
        ]
        
        return {
            "new_episodes": new_episodes,
            "is_new_feed": False,
            "feed_info": rss_data['feed_info'],
            "total_episodes": len(rss_data['episodes']),
            "new_episodes_count": len(new_episodes),
            "already_processed": len(processed_guids)
        }

# --- CONTENT GENERATOR ---
class ContentGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.llm_analytical = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def create_episode_brief(self, transcript_content: str, episode_title: str = None) -> Dict:
        """Create episode brief from podcast transcript"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content producer creating episode briefs. Based on transcript, create:
            
            1. Episode Summary (2-3 sentences)
            2. Key Topics Discussed (bullet points)
            3. Notable Quotes (2-3 impactful quotes)
            4. Key Takeaways (3-5 main points)
            5. Target Audience
            
            Format as a clean, readable brief."""),
            ("human", """Episode Title: {episode_title}
            
            Transcript Content:
            {transcript_content}
            
            Create an episode brief.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "episode_title": episode_title or "Untitled Episode",
            "transcript_content": transcript_content[:8000]
        })
        
        return {
            "success": True,
            "content": response.content,
            "type": "episode_brief"
        }
    
    def generate_summary(self, content: str, summary_type: str = "executive", length: str = "medium") -> Dict:
        """Generate different types of summaries"""
        
        length_guidelines = {
            "short": "1-2 sentences",
            "medium": "1 paragraph (100-150 words)", 
            "long": "2-3 paragraphs (200-300 words)"
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a professional content summarizer. Create a {{summary_type}} summary that is {{length_guidelines.get(length, 'medium length')}}.
                    
            Guidelines:
            - Short: {{length_guidelines['short']}}
            - Medium: {{length_guidelines['medium']}}  
            - Long: {{length_guidelines['long']}}
            """),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm_analytical
        response = chain.invoke({"content": content})
        
        return {
            "success": True,
            "content": response.content,
            "type": "summary"
        }

rss_processor = EnhancedRSSProcessor()
content_generator = ContentGenerator()
processing_status = {}

# --- UTILS ---
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_dynamic_splitter(text_content: str):
    total_chars = len(text_content)
    
    if total_chars < 2000:
        size = total_chars
        overlap = 0
    else:
        size = min(max(int(total_chars * 0.15), 800), 2500)
        overlap = int(size * 0.15)

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

# ✅ UPDATED: Speaker-aware PDF generation with Supabase upload
def transcript_to_pdf_with_speakers(transcript_obj, filename: str, upload_to_supabase: bool = True) -> Dict:
    """Create PDF with speaker diarization labels and upload to Supabase"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)  # Add proper margins
    pdf.set_auto_page_break(auto=True, margin=15)  # Auto page break
    pdf.set_font("Arial", size=10)
    
    # Add title
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, txt=f"Transcript: {filename}")
    pdf.ln(5)
    
    # Check if speaker labels are available
    if hasattr(transcript_obj, 'utterances') and transcript_obj.utterances:
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, txt=f"Total Speakers Detected: {len(set(u.speaker for u in transcript_obj.utterances))}")
        pdf.ln(3)
        
        # Format with speaker labels
        for utterance in transcript_obj.utterances:
            # Speaker label in bold
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 6, txt=f"\n{utterance.speaker}:")
            
            # Speaker text
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 6, txt=utterance.text)
    else:
        # Fallback to plain text if no speaker labels
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, txt=transcript_obj.text)
    
    # Save to temporary file first
    temp_pdf_path = os.path.join(TEMP_TRANSCRIPT_DIR, f"{filename}.pdf")
    pdf.output(temp_pdf_path)
    
    result = {
        "local_path": temp_pdf_path,
        "supabase_uploaded": False
    }
    
    # Upload to Supabase if requested
    if upload_to_supabase:
        upload_result = supabase_manager.upload_pdf(temp_pdf_path, f"{filename}.pdf")
        if upload_result["success"]:
            result["supabase_uploaded"] = True
            result["supabase_url"] = upload_result["url"]
            result["supabase_path"] = upload_result["path"]
            
            # Clean up local temp file after successful upload
            try:
                os.remove(temp_pdf_path)
            except:
                pass
    
    return result

# ✅ UPDATED: Format transcript with speaker labels for embedding
def format_transcript_with_speakers(transcript_obj) -> str:
    """Format transcript text with speaker labels for better context"""
    if hasattr(transcript_obj, 'utterances') and transcript_obj.utterances:
        formatted_text = ""
        for utterance in transcript_obj.utterances:
            formatted_text += f"\n{utterance.speaker}: {utterance.text}\n"
        return formatted_text
    else:
        # Fallback to plain text
        return transcript_obj.text

def transcript_to_pdf(text: str, filename: str, upload_to_supabase: bool = True) -> Dict:
    """Legacy function for simple text to PDF with Supabase upload"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)  # Add proper margins
    pdf.set_auto_page_break(auto=True, margin=15)  # Auto page break
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    
    temp_pdf_path = os.path.join(TEMP_TRANSCRIPT_DIR, f"{filename}.pdf")
    pdf.output(temp_pdf_path)
    
    result = {
        "local_path": temp_pdf_path,
        "supabase_uploaded": False
    }
    
    if upload_to_supabase:
        upload_result = supabase_manager.upload_pdf(temp_pdf_path, f"{filename}.pdf")
        if upload_result["success"]:
            result["supabase_uploaded"] = True
            result["supabase_url"] = upload_result["url"]
            result["supabase_path"] = upload_result["path"]
            
            # Clean up local temp file
            try:
                os.remove(temp_pdf_path)
            except:
                pass
    
    return result

def save_text_to_pdf(text: str, output_path: str):
    """Saves transcription text into a professional PDF format."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)  # Add proper margins
    pdf.set_auto_page_break(auto=True, margin=15)  # Auto page break
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    pdf.output(output_path)

def load_vectorstore():
    """Load Pinecone vector store"""
    try:
        index = get_pinecone_index()
        if index:
            return PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
        return None
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def get_all_file_links(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
    links = []
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        content_type = response.headers.get('Content-Type', '').lower()

        feed = feedparser.parse(response.content)
        if feed.entries:
            for entry in feed.entries:
                if hasattr(entry, 'link'): links.append(entry.link)
                if hasattr(entry, 'enclosures'):
                    for enc in entry.enclosures:
                        links.append(enc.href)
            return list(set(links))

        if "sitemap" in url.lower() or "xml" in content_type:
            soup = BeautifulSoup(response.content, 'xml')
            sitemap_links = [loc.text for loc in soup.find_all('loc')]
            if sitemap_links:
                return list(set(sitemap_links))

        soup = BeautifulSoup(response.content, 'html.parser')
        extensions = ('.pdf', '.mp3', '.wav', '.m4a')
        for a in soup.find_all('a', href=True):
            full_url = urljoin(url, a['href'])
            if any(full_url.lower().endswith(ext) for ext in extensions):
                links.append(full_url)

        feed = feedparser.parse(response.content)
        if feed.entries:
            for entry in feed.entries:
                if hasattr(entry, 'link'): links.append(entry.link)
            return list(set(links))
        
        return list(set(links)) if links else [url]

    except Exception as e:
        print(f"Error scanning URL: {e}")
        return [url]
    
def save_batch_to_vectorstore(chunks):
    """Helper to save increments to Pinecone."""
    if not chunks: return
    vs = load_vectorstore()
    if vs is None:
        vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
    else:
        vs.add_documents(chunks)

def process_in_batches_task(file_links: list, task_id: str):
    temp_dir = tempfile.mkdtemp()
    batch_size = 5
    processed_count = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    processing_status[task_id]["errors"] = []

    for i in range(0, len(file_links), batch_size):
        current_batch = file_links[i : i + batch_size]
        all_chunks_in_batch = []

        for link in current_batch:
            try:
                parsed_path = urlparse(link).path
                ext = parsed_path.split('.')[-1].lower() if '.' in parsed_path else ""
                unique_suffix = uuid.uuid4().hex[:4]
                base_name = os.path.basename(parsed_path) or f"web_{unique_suffix}"
                text = ""
                supabase_pdf_url = None

                if ext in ["mp3", "wav", "m4a"]:
                    # Download to temp, transcribe, then upload to Supabase
                    temp_audio_path = os.path.join(temp_dir, f"{base_name}.{ext}")
                    
                    try:
                        resp = requests.get(link, timeout=120, stream=True)
                        with open(temp_audio_path, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Upload audio to Supabase
                        audio_upload = supabase_manager.upload_audio(temp_audio_path, f"{base_name}.{ext}")
                        
                        # Transcribe with speaker diarization
                        config = aai.TranscriptionConfig(speaker_labels=True)
                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(temp_audio_path, config=config)
                        
                        if transcript.status != aai.TranscriptStatus.error:
                            # Format with speaker labels
                            text = format_transcript_with_speakers(transcript)
                            
                            # Create and upload PDF to Supabase
                            pdf_result = transcript_to_pdf_with_speakers(transcript, base_name, upload_to_supabase=True)
                            if pdf_result.get("supabase_uploaded"):
                                supabase_pdf_url = pdf_result["supabase_url"]
                        
                        # Clean up temp audio
                        os.remove(temp_audio_path)
                        
                    except Exception as audio_err:
                        print(f"Audio processing error: {audio_err}")
                
                elif ext == 'pdf':
                    resp = requests.get(link, timeout=30)
                    local_path = os.path.join(temp_dir, f"{base_name}.pdf")
                    with open(local_path, "wb") as f: 
                        f.write(resp.content)
                    
                    text = extract_text_from_pdf(local_path)
                    
                    # Upload PDF to Supabase
                    pdf_upload = supabase_manager.upload_pdf(local_path, f"{base_name}.pdf")
                    if pdf_upload["success"]:
                        supabase_pdf_url = pdf_upload["url"]
                    
                    # Clean up temp file
                    os.remove(local_path)

                else:
                    resp = requests.get(link, timeout=15)
                    soup = BeautifulSoup(resp.content, 'lxml')
                    for s in soup(["script", "style"]): s.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    
                    if text.strip():
                        pdf_result = transcript_to_pdf(text, f"{base_name}_web", upload_to_supabase=True)
                        if pdf_result.get("supabase_uploaded"):
                            supabase_pdf_url = pdf_result["supabase_url"]

                if text.strip():
                    splitter = get_dynamic_splitter(text)
                    metadata = {
                        "source": link,
                        "supabase_pdf_url": supabase_pdf_url
                    }
                    all_chunks_in_batch.extend(splitter.create_documents([text], metadatas=[metadata]))

            except Exception as e:
                error_msg = f"Failed {link}: {str(e)}"
                processing_status[task_id]["errors"].append(error_msg)
                print(error_msg)
            
            processed_count += 1
            processing_status[task_id]["current"] = processed_count

        if all_chunks_in_batch:
            save_batch_to_vectorstore(all_chunks_in_batch)
        gc.collect()

    processing_status[task_id]["status"] = "completed"
    shutil.rmtree(temp_dir, ignore_errors=True)

# --- TOOLS ---
@tool
def search_documents(query: str):
    """
    Search the document database for specific facts, context, or summaries.
    Use this for any questions regarding uploaded documents, technical data, or transcripts.
    """
    vs = load_vectorstore()
    if not vs:
        return "No documents have been uploaded yet."
    
    docs = vs.similarity_search(query, k=20)  
    context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])
    return context

tools = [search_documents]
tool_node = ToolNode(tools)

# --- AGENT 1: GREETING AGENT ---
def greeting_agent(state: AgentState):
    """Handles only greetings like hello, hi, etc."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Respond ONLY to greetings like hello, hi, good morning, etc. Keep responses brief and polite."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"history": state["messages"], "question": state["question"]})
    return {
        "answer": response.content,
        "sources": [],
        "episode_titles": []
    }

# --- AGENT 2: IRRELEVANT QUERY AGENT ---
def irrelevant_query_agent(state: AgentState):
    """Handles irrelevant queries using ChatGPT search."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. The user asked a question that is not related to podcast production, hospitality, or the documents in our system. 
        Please provide a helpful, accurate answer to their question. Be conversational and informative.
        
        If the question is about:
        - General knowledge, news, facts → Provide accurate information
        - Personal advice → Give helpful suggestions
        - Technical topics → Explain clearly
        - Any other topic → Be helpful and informative
        
        Keep responses natural and conversational."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"history": state["messages"], "question": state["question"]})
    return {
        "answer": response.content,
        "sources": [],
        "episode_titles": []
    }

# --- AGENT 3: GENERAL CHAT AGENT ---
def general_chat_agent(state: AgentState):
    """Handles general conversation related to podcast/hospitality but not requiring documents."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant knowledgeable about podcast production, hospitality industry, and content creation. 
        Engage in natural conversation about these topics. For specific factual information or document details, suggest using the document search."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"history": state["messages"], "question": state["question"]})
    return {
        "answer": response.content,
        "sources": [],
        "episode_titles": []
    }

# --- AGENT 4: CONTEXT AGENT (RAG) ---
def context_agent(state: AgentState):
    """Handles retrieval and task-based generation (blogs, summaries, etc.)"""
    print("--- DEBUG: Entered context_agent ---")
    llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_main = ChatOpenAI(model="gpt-4o", temperature=0.5)

    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"][-3:]])
    rewrite_prompt = f"""
    Analyze the history and user question. Identify the main SUBJECT for searching.
    
    IMPORTANT: Extract the core topic/subject that should be searched in documents.
    If user asks about "Neighborhood Gathering", search for "neighborhood gathering" or "community" or similar terms.
    If user asks about "AI and ML", search for "artificial intelligence" or "machine learning".
    If user asks about specific topics, extract those exact keywords.
    
    Ignore task words like 'write a blog', 'tell me more', 'create content', etc.
    
    History: {history_str}
    User Question: {state['question']}
    
    Search Keywords (exact topic to search):"""

    search_query = llm_fast.invoke(rewrite_prompt).content
    print(f"--- DEBUG: Rewriter found topic: {search_query} ---")

    # Load vectorstore at the beginning
    vs = load_vectorstore()
    
    # First, get retrieved documents to extract actual episode titles
    retrieved_episode_titles = []
    temp_docs = []
    if vs:
        # Get documents first to extract episode titles
        temp_docs = vs.similarity_search(search_query, k=6)
        for doc in temp_docs:
            ep_title = doc.metadata.get('episode_title', '')
            if ep_title and ep_title != 'Feed Information':
                retrieved_episode_titles.append(ep_title)
    
    # Extract podcast title and context from user query AND retrieved documents
    extraction_prompt = f"""
    Analyze the user's query and the retrieved episode titles to extract:
    1. Podcast episode title (if mentioned in query OR use most relevant from retrieved)
    2. Main topic/theme
    3. Guest information (if mentioned)
    4. Key focus areas
    
    User Query: {state['question']}
    Retrieved Episode Titles: {', '.join(retrieved_episode_titles[:3])}
    
    IMPORTANT: If user mentions a specific episode title in their query, use that. 
    If not, use the most relevant episode title from the retrieved titles.
    
    Respond in this format:
    TITLE: [extracted title or "Untitled Episode"]
    TOPIC: [main topic/theme]
    GUEST: [guest name or "Not specified"]
    FOCUS: [key focus areas]
    """
    
    extracted_info = llm_fast.invoke(extraction_prompt).content
    print(f"--- DEBUG: Extracted info: {extracted_info} ---")
    
    # Parse extracted info
    lines = extracted_info.split('\n')
    title = "Untitled Episode"
    topic = ""
    guest = "Not specified"
    focus = ""
    
    for line in lines:
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()
        elif line.startswith("TOPIC:"):
            topic = line.replace("TOPIC:", "").strip()
        elif line.startswith("GUEST:"):
            guest = line.replace("GUEST:", "").strip()
        elif line.startswith("FOCUS:"):
            focus = line.replace("FOCUS:", "").strip()
    
    # Override title if we have retrieved episode titles and user didn't specify one
    if not title or title == "Untitled Episode" and retrieved_episode_titles:
        title = retrieved_episode_titles[0]  # Use most relevant retrieved title
        print(f"--- DEBUG: Using retrieved episode title: {title} ---")
    
    # Initialize pinecone_context early
    pinecone_context = ""
    
    # Dynamic Master Prompt based on retrieved content
    # Analyze retrieved documents to understand the podcast context
    podcast_analysis_prompt = f"""
    Based on the retrieved document context below, identify:
    1. What type of podcast/content this appears to be (hospitality, tech, lifestyle, etc.)
    2. The general tone and style of the content
    3. Target audience based on the content
    4. Any specific format or structure patterns visible
    
    Retrieved Context Sample: {pinecone_context[:1000] if pinecone_context else "No content found"}
    
    Provide a brief analysis of the podcast type and style:
    """
    
    podcast_style = "general conversational podcast"
    try:
        style_analysis = llm_fast.invoke(podcast_analysis_prompt)
        podcast_style = style_analysis.content.strip()
    except:
        podcast_style = "engaging conversational content"
    
    # Dynamic Master Prompt
    master_prompt_context = f"""
    You are a skilled content creator and producer tasked with creating compelling content based on the user's request.
    
    User Request Analysis:
    - Requested Topic: {topic}
    - Episode Title: {title}
    - Guest: {guest}
    - Focus Areas: {focus}
    
    Content Style Context: {podcast_style}
    
    Your task is to create high-quality, engaging content that matches the user's request and the identified content style.
    
    REQUIREMENTS:
    ● Create content that is engaging and well-structured
    ● Match the tone and style appropriate for the topic and audience
    ● Include relevant sections, questions, or discussion points
    ● Ensure the content is practical and actionable
    ● Maintain a professional yet conversational approach
    ● When available, reference specific speakers and their contributions from the transcript
    
    CONTENT STRUCTURE:
    ● Clear introduction that sets the context
    ● Well-organized main sections with logical flow
    ● Thoughtful questions or discussion points
    ● Appropriate conclusion and takeaways
    ● Any additional elements relevant to the content type
    
    TONE AND STYLE:
    - Professional yet conversational and accessible
    - Engaging and insightful
    - Adapted to the specific topic and audience
    - Clear, well-structured, and practical
    """

    final_docs = []
    if vs:
        # Try multiple search strategies for better results
        search_queries = [search_query.strip()]
        
        # Generate alternative search terms dynamically using LLM
        alt_search_prompt = f"""
        Based on the search query "{search_query}", generate 3-4 alternative search terms that might find relevant content.
        Think about synonyms, related concepts, and different ways this topic might be discussed.
        Return only the search terms, one per line, no explanations.
        
        Examples:
        - Query: "neighborhood gathering" → "community events", "local meetups", "neighbors", "community"
        - Query: "artificial intelligence" → "machine learning", "AI technology", "automation", "smart systems"
        """
        
        try:
            alt_terms_response = llm_fast.invoke(alt_search_prompt)
            alt_terms = [term.strip() for term in alt_terms_response.content.split('\n') if term.strip()]
            search_queries.extend(alt_terms[:4])  # Limit to 4 alternatives
        except:
            # Fallback to basic variations if LLM fails
            base_terms = search_query.lower().split()
            for term in base_terms:
                if len(term) > 3:
                    search_queries.append(term)
        
        all_docs = []
        for query in search_queries[:3]:  # Limit to avoid too many searches
            try:
                docs = vs.similarity_search(query, k=3)
                all_docs.extend(docs)
            except:
                continue
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Use top most relevant docs
        final_docs = unique_docs[:6]
        pinecone_context = "\n\n".join([d.page_content for d in final_docs])
        
        # Add debug info about what was searched
        print(f"--- DEBUG: Searched for: {search_queries[:3]}, Found {len(final_docs)} relevant docs ---")

    # Combine both contexts
    combined_context = f"""
    === THE CURIOUS CONCIERGE MASTER PROMPT ===
    {master_prompt_context}
    
    === RETRIEVED DOCUMENT CONTEXT (WITH SPEAKER LABELS) ===
    {pinecone_context if pinecone_context else "No specific documents found for this query."}
    
    === USER ORIGINAL QUERY ===
    {state['question']}
    """

    system_instruction = """
    You are The Curious Concierge podcast producer with access to both retrieved documents and the master prompt template.
    1. USE the master prompt structure exactly as specified
    2. INCORPORATE relevant information from retrieved documents when available
    3. PAY ATTENTION to speaker labels (Speaker A, Speaker B, etc.) in the transcript to attribute quotes and ideas correctly
    4. FOLLOW the exact format with emojis, sections, and question counts
    5. MAINTAIN the curious, insightful, cinematic tone
    6. CREATE a complete, ready-to-use episode brief
    7. When referencing conversations, mention which speaker said what if that information is available
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("system", f"COMBINED CONTEXT:\n{combined_context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm_main
    response = chain.invoke({
        "history": state["messages"], 
        "question": state["question"]
    })

    # Extract source information and episode titles from retrieved documents
    source_info = []
    episode_titles = set()
    
    # Use final_docs that contain the actual retrieved documents
    if final_docs:
        for doc in final_docs:
            # Extract episode title from metadata
            episode_title = doc.metadata.get('episode_title', '')
            if episode_title and episode_title != 'Feed Information':
                episode_titles.add(episode_title)
            
            # Extract source information
            source = doc.metadata.get('source', '')
            feed_title = doc.metadata.get('feed_title', '')
            content_type = doc.metadata.get('content_type', '')
            supabase_pdf_url = doc.metadata.get('supabase_pdf_url', '')
            supabase_audio_url = doc.metadata.get('supabase_audio_url', '')
            
            if source:
                source_info.append({
                    'source': source,
                    'feed_title': feed_title,
                    'episode_title': episode_title,
                    'content_type': content_type,
                    'supabase_pdf_url': supabase_pdf_url,
                    'supabase_audio_url': supabase_audio_url
                })
        
        # Add debug info for sources
        print(f"--- DEBUG: Sources found: {len(source_info)} ---")
        for i, src in enumerate(source_info[:3]):  # Show first 3 sources
            print(f"--- SOURCE {i+1}: {src['episode_title']} from {src['feed_title']} ---")
    elif temp_docs:
        # Fallback to temp_docs if final_docs not available
        for doc in temp_docs:
            episode_title = doc.metadata.get('episode_title', '')
            if episode_title and episode_title != 'Feed Information':
                episode_titles.add(episode_title)
            
            source = doc.metadata.get('source', '')
            feed_title = doc.metadata.get('feed_title', '')
            content_type = doc.metadata.get('content_type', '')
            supabase_pdf_url = doc.metadata.get('supabase_pdf_url', '')
            supabase_audio_url = doc.metadata.get('supabase_audio_url', '')
            
            if source:
                source_info.append({
                    'source': source,
                    'feed_title': feed_title,
                    'episode_title': episode_title,
                    'content_type': content_type,
                    'supabase_pdf_url': supabase_pdf_url,
                    'supabase_audio_url': supabase_audio_url
                })
        
        print(f"--- DEBUG: Fallback sources: {len(source_info)} ---")
    
    return {
        "answer": response.content, 
        "context": combined_context,
        "sources": source_info,
        "episode_titles": list(episode_titles) if episode_titles else []
    }

# --- ROUTER LOGIC ---
def route_question(state: AgentState) -> Literal["greeting", "irrelevant", "general_chat", "context"]:
    """Determines which agent should handle the request."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    Analyze the user input and categorize it into one of these types:
    
    1. "greeting" - Simple greetings like hello, hi, good morning, hey, etc.
    2. "context" - Questions that require searching documents, creating episode briefs, summarizing content, or specific factual information from uploaded materials

    3. "general_chat" - General conversation about podcast production, hospitality industry in Asia, content creation, but not requiring specific document search
    
    4. "irrelevant" - Questions that are NOT specifically about:
    - Creating podcast episodes/briefs for The Curious Concierge
    - Analyzing uploaded transcripts or documents
    - Hospitality industry in Asia context
    - Specific guests or episodes mentioned in documents
    Examples: "make reels on AI", "weather today", "sports scores", "general tech topics", "personal advice"
    
    IMPORTANT: If user is asking for general content creation (reels, blogs, etc.) about topics like AI/ML without mentioning hospitality or The Curious Concierge specifically, classify as "irrelevant".
    
    User Input: {state['question']}
    
    Respond with ONLY one word: greeting, irrelevant, general_chat, or context
    Decision:
    """
    result = llm.invoke(prompt)
    decision = result.content.strip().lower()
    
    # Map by decision to the correct agent
    print(f"--- DEBUG: Router decision: {decision} ---")
    if "greeting" in decision:
        print("--- DEBUG: Routing to greeting_agent ---")
        return "greeting"
    elif "irrelevant" in decision:
        print("--- DEBUG: Routing to irrelevant_query_agent ---")
        return "irrelevant"
    elif "general_chat" in decision:
        print("--- DEBUG: Routing to general_chat_agent ---")
        return "general_chat"
    else:
        print("--- DEBUG: Routing to context_agent (default) ---")
        return "context"

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("greeting_agent", greeting_agent)
workflow.add_node("context_agent", context_agent)
workflow.add_node("general_chat_agent", general_chat_agent)
workflow.add_node("irrelevant_query_agent", irrelevant_query_agent)
workflow.add_node("tools", tool_node)

workflow.set_conditional_entry_point(
    route_question,
    {
        "greeting": "greeting_agent",
        "context": "context_agent",
        "general_chat": "general_chat_agent",
        "irrelevant": "irrelevant_query_agent"
    }
)

workflow.add_edge("greeting_agent", END)
workflow.add_edge("context_agent", END)
workflow.add_edge("general_chat_agent", END)
workflow.add_edge("irrelevant_query_agent", END)
workflow.add_edge("tools", "context_agent")

app_agent = workflow.compile()

# --- DEDUPLICATED BACKGROUND TASK FOR MULTIPLE FEEDS ---
def process_multiple_feeds_task(feed_urls: List[str], task_id: str):
    """Process multiple RSS feeds one by one with audio transcription, speaker diarization, and Supabase storage"""
    temp_dir = tempfile.mkdtemp()
    
    processing_status[task_id]["errors"] = []
    processing_status[task_id]["success"] = 0  # Transcribed episodes
    processing_status[task_id]["metadata"] = 0  # Metadata only episodes
    processing_status[task_id]["unsuccessful"] = 0  # Failed episodes
    
    for idx, feed_url in enumerate(feed_urls):
        try:
            processing_status[task_id]["current"] = idx + 1
            processing_status[task_id]["current_feed"] = feed_url
            
            result = rss_processor.get_new_episodes(feed_url)
            
            if result.get("error"):
                error_msg = f"Feed {feed_url}: {result['error']}"
                processing_status[task_id]["errors"].append(error_msg)
                processing_status[task_id]["results"].append({
                    "feed_url": feed_url,
                    "success": False,
                    "error": result['error']
                })
                continue
            
            new_episodes = result["new_episodes"]
            is_new_feed = result["is_new_feed"]
            feed_info = result["feed_info"]
            
            if not new_episodes and not is_new_feed:
                processing_status[task_id]["results"].append({
                    "feed_url": feed_url,
                    "success": True,
                    "message": "Already processed, no new episodes",
                    "feed_title": feed_info.get('title', 'Unknown'),
                    "already_processed": result.get("already_processed", 0),
                    "new_episodes": 0
                })
                continue
            
            if is_new_feed:
                metadata_manager.add_rss_feed(feed_url, feed_info.get('title', 'Unknown Feed'))
            
            if feed_info.get('description'):
                feed_content = f"Podcast: {feed_info.get('title', 'Unknown')}\nDescription: {feed_info['description']}\nFeed URL: {feed_url}"
                splitter = get_dynamic_splitter(feed_content)
                feed_chunks = splitter.create_documents([feed_content], metadatas=[{
                    "source": feed_url,
                    "episode_title": "Feed Information",
                    "feed_url": feed_url,
                    "feed_title": feed_info.get('title', 'Unknown'),
                    "content_type": "feed_description"
                }])
                
                vs = load_vectorstore()
                if vs is None:
                    vs = PineconeVectorStore.from_documents(feed_chunks, embeddings, index_name=PINECONE_INDEX_NAME)
                else:
                    vs.add_documents(feed_chunks)
            
            episodes_processed = 0
            for episode in new_episodes:
                audio_path = None
                supabase_audio_url = None
                supabase_pdf_url = None
                max_retries = 3
                retry_count = 0
                
                try:
                    audio_url = episode['audio_url']
                    episode_title = episode['title']
                    
                    if audio_url and any(audio_url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a']):
                        print(f"📥 Downloading: {episode_title} from {feed_info.get('title', 'Unknown')}")
                        
                        while retry_count < max_retries:
                            try:
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                }
                                response = requests.get(audio_url, timeout=120, headers=headers, stream=True)
                                response.raise_for_status()
                                
                                audio_filename = f"episode_{uuid.uuid4().hex[:8]}.mp3"
                                audio_path = os.path.join(temp_dir, audio_filename)
                                
                                with open(audio_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                
                                print(f"✅ Downloaded: {episode_title}")
                                
                                # Upload audio to Supabase
                                audio_upload = supabase_manager.upload_audio(audio_path, audio_filename)
                                if audio_upload["success"]:
                                    supabase_audio_url = audio_upload["url"]
                                    print(f"☁️ Audio uploaded to Supabase: {episode_title}")
                                
                                break
                                
                            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    raise Exception(f"Download failed after {max_retries} attempts: {str(e)}")
                                print(f"⚠️ Retry {retry_count}/{max_retries} for: {episode_title}")
                                import time
                                time.sleep(5 * retry_count)
                        
                        # Transcribe with speaker diarization
                        print(f"🎙️ Transcribing with Speaker Diarization: {episode_title}")
                        transcription_success = False
                        transcription_retries = 3
                        transcription_attempt = 0
                        
                        while transcription_attempt < transcription_retries and not transcription_success:
                            try:
                                print(f"🎙️ Transcription attempt {transcription_attempt + 1}/{transcription_retries}")
                                
                                config = aai.TranscriptionConfig(speaker_labels=True)
                                transcriber = aai.Transcriber()
                                transcript = transcriber.transcribe(audio_path, config=config)
                                
                                poll_count = 0
                                max_polls = 200
                                
                                while poll_count < max_polls:
                                    try:
                                        if transcript.status == aai.TranscriptStatus.completed:
                                            transcription_success = True
                                            break
                                        elif transcript.status == aai.TranscriptStatus.error:
                                            raise Exception(f"Transcription API error: {transcript.error}")
                                        
                                        import time
                                        time.sleep(3)
                                        poll_count += 1
                                        transcript = transcriber.get_transcript(transcript.id)
                                        
                                    except Exception as poll_err:
                                        print(f"⚠️ Poll error: {str(poll_err)[:50]}")
                                        raise
                                
                                if not transcription_success and poll_count >= max_polls:
                                    raise Exception("Transcription timeout")
                                
                                if transcription_success and transcript.text:
                                    # Format transcript with speaker labels
                                    formatted_transcript = format_transcript_with_speakers(transcript)
                                    
                                    # Get speaker count
                                    speaker_count = 0
                                    if hasattr(transcript, 'utterances') and transcript.utterances:
                                        speaker_count = len(set(u.speaker for u in transcript.utterances))
                                    
                                    print(f"👥 Detected {speaker_count} speakers in: {episode_title}")
                                    
                                    content = f"Podcast: {feed_info.get('title', 'Unknown')}\n"
                                    content += f"Episode: {episode_title}\n"
                                    content += f"Speakers Detected: {speaker_count}\n"
                                    
                                    if episode.get('description'):
                                        content += f"Description: {episode['description']}\n"
                                    
                                    content += f"\nTranscript with Speaker Labels:\n{formatted_transcript}"
                                    
                                    safe_filename = "".join(c for c in episode_title if c.isalnum() or c in (' ', '-', '_')).strip()
                                    safe_filename = safe_filename[:100]
                                    
                                    # Save PDF to Supabase
                                    pdf_result = transcript_to_pdf_with_speakers(
                                        transcript,
                                        f"{safe_filename}_{uuid.uuid4().hex[:4]}",
                                        upload_to_supabase=True
                                    )
                                    
                                    if pdf_result.get("supabase_uploaded"):
                                        supabase_pdf_url = pdf_result["supabase_url"]
                                        print(f"☁️ PDF uploaded to Supabase: {episode_title}")
                                    
                                    splitter = get_dynamic_splitter(content)
                                    chunks = splitter.create_documents([content], metadatas=[{
                                        "source": audio_url,
                                        "episode_title": episode_title,
                                        "feed_url": feed_url,
                                        "feed_title": feed_info.get('title', 'Unknown'),
                                        "content_type": "episode_transcript",
                                        "speaker_count": speaker_count,
                                        "supabase_audio_url": supabase_audio_url,
                                        "supabase_pdf_url": supabase_pdf_url
                                    }])
                                    
                                    vs = load_vectorstore()
                                    if vs is None:
                                        vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
                                    else:
                                        vs.add_documents(chunks)
                                    
                                    print(f"✅ Transcribed & Stored: {episode_title} ({len(chunks)} chunks, {speaker_count} speakers)")
                                    processing_status[task_id]["success"] += 1  # Increment transcribed counter
                                    break
                                    
                            except Exception as transcribe_err:
                                transcription_attempt += 1
                                error_detail = str(transcribe_err)
                                
                                if "getaddrinfo failed" in error_detail or "Connection" in error_detail:
                                    print(f"⚠️ Network error (Attempt {transcription_attempt}/{transcription_retries}): {error_detail[:100]}")
                                    
                        if episode.get('description'):
                            content += f"Description: {episode['description']}\n"

                        content += f"\nTranscript with Speaker Labels:\n{formatted_transcript}"

                        safe_filename = "".join(c for c in episode_title if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_filename = safe_filename[:100]

                        # Save PDF to Supabase
                        pdf_result = transcript_to_pdf_with_speakers(
                            transcript,
                            f"{safe_filename}_{uuid.uuid4().hex[:4]}",
                            upload_to_supabase=True
                        )

                        if pdf_result.get("supabase_uploaded"):
                            supabase_pdf_url = pdf_result["supabase_url"]
                            print(f" PDF uploaded to Supabase: {episode_title}")

                            content += f"\nDescription: {episode['description']}"
                        
                        splitter = get_dynamic_splitter(content)
                        chunks = splitter.create_documents([content], metadatas=[{
                            "source": audio_url,
                            "episode_title": episode_title,
                            "feed_url": feed_url,
                            "feed_title": feed_info.get('title', 'Unknown'),
                            "content_type": "episode_metadata"
                        }])
                        
                        vs = load_vectorstore()
                        if vs is None:
                            vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
                        else:
                            vs.add_documents(chunks)
                        
                        processing_status[task_id]["metadata"] += 1  # Increment metadata counter
                    
                    metadata_manager.add_episode(
                        feed_url, 
                        episode['guid'], 
                        episode_title,
                        audio_url=supabase_audio_url,
                        pdf_url=supabase_pdf_url
                    )
                    episodes_processed += 1
                    
                except Exception as e:
                    error_msg = f"Episode error in {feed_url}: {str(e)}"
                    processing_status[task_id]["errors"].append(error_msg)
                    processing_status[task_id]["unsuccessful"] += 1  # Increment failed counter
                
                finally:
                    # Clean up temp audio file
                    if audio_path and os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                
                if episodes_processed < len(new_episodes):
                    import time
                    time.sleep(2)
            
            processing_status[task_id]["feeds_processed"] += 1
            processing_status[task_id]["results"].append({
                "feed_url": feed_url,
                "success": True,
                "feed_title": feed_info.get('title', 'Unknown'),
                "new_episodes": len(new_episodes),
                "episodes_processed": episodes_processed,
                "is_new_feed": is_new_feed
            })
            
            import time
            time.sleep(1)
            
            if idx + 1 < len(feed_urls):
                time.sleep(3)
            
        except Exception as e:
            error_msg = f"Failed processing {feed_url}: {str(e)}"
            processing_status[task_id]["errors"].append(error_msg)
            processing_status[task_id]["results"].append({
                "feed_url": feed_url,
                "success": False,
                "error": str(e)
            })
    
    processing_status[task_id]["status"] = "completed"
    shutil.rmtree(temp_dir, ignore_errors=True)

def process_deduplicated_episodes_task(feed_url: str, episodes: List[Dict], feed_info: Dict, task_id: str):
    """Process episodes with deduplication tracking and Supabase storage"""
    processed_count = 0
    
    processing_status[task_id]["errors"] = []
    
    if feed_info.get('description'):
        feed_content = f"Podcast: {feed_info.get('title', 'Unknown')}\nDescription: {feed_info['description']}\nFeed URL: {feed_url}"
        splitter = get_dynamic_splitter(feed_content)
        feed_chunks = splitter.create_documents([feed_content], metadatas=[{
            "source": feed_url,
            "episode_title": "Feed Information",
            "feed_url": feed_url,
            "feed_title": feed_info.get('title', 'Unknown'),
            "content_type": "feed_description"
        }])
        
        vs = load_vectorstore()
        if vs is None:
            vs = PineconeVectorStore.from_documents(feed_chunks, embeddings, index_name=PINECONE_INDEX_NAME)
        else:
            vs.add_documents(feed_chunks)
    
    for episode in episodes:
        audio_path = None
        episode_success = False
        supabase_audio_url = None
        supabase_pdf_url = None
        
        try:
            audio_url = episode.get('audio_url', '').strip()
            episode_title = episode.get('title', 'Untitled Episode')
            
            if not audio_url:
                print(f"⚠️ Skipping {episode_title}: No audio URL found")
                processing_status[task_id]["unsuccessful"] += 1
                processing_status[task_id]["errors"].append(f"No audio URL: {episode_title}")
                processed_count += 1
                processing_status[task_id]["current"] = processed_count
                continue
            
            try:
                parsed = urlparse(audio_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError("Invalid URL format")
            except Exception:
                print(f"⚠️ Skipping {episode_title}: Invalid URL format")
                processing_status[task_id]["errors"].append(f"Invalid URL: {episode_title}")
                processing_status[task_id]["unsuccessful"] += 1
                processed_count += 1
                processing_status[task_id]["current"] = processed_count
                continue
            
            is_audio_url = any(audio_url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a'])
            
            if is_audio_url:
                print(f"📥 Starting: {episode_title[:50]}...")
                
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    head_response = requests.head(audio_url, timeout=5, headers=headers, allow_redirects=True)
                    
                    if head_response.status_code >= 400:
                        raise Exception(f"Status {head_response.status_code}")
                    
                    print(f"✅ URL OK: {episode_title[:50]}")
                    
                except Exception as head_err:
                    print(f"⚠️ URL failed: {episode_title[:50]} - {str(head_err)[:50]}")
                    processing_status[task_id]["errors"].append(f"URL inaccessible: {episode_title}")
                    processing_status[task_id]["unsuccessful"] += 1
                    processed_count += 1
                    processing_status[task_id]["current"] = processed_count
                    continue
                
                max_retries = 2
                retry_count = 0
                download_success = False
                
                while retry_count < max_retries and not download_success:
                    try:
                        print(f"⬇️ Downloading {retry_count + 1}/{max_retries}: {episode_title[:40]}")
                        
                        response = requests.get(
                            audio_url,
                            timeout=240,
                            headers=headers,
                            stream=True,
                            allow_redirects=True
                        )
                        response.raise_for_status()
                        
                        safe_filename = "".join(c for c in episode_title if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_filename = safe_filename[:80] or "episode"
                        audio_filename = f"{safe_filename}_{uuid.uuid4().hex[:6]}.mp3"
                        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
                        
                        total_size = 0
                        with open(audio_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=16384):
                                if chunk:
                                    f.write(chunk)
                                    total_size += len(chunk)
                        
                        if total_size < 1024:
                            raise Exception("File too small")
                        
                        print(f"✅ Downloaded: {total_size / 1024 / 1024:.2f} MB")
                        
                        # Upload audio to Supabase
                        audio_upload = supabase_manager.upload_audio(audio_path, audio_filename)
                        if audio_upload["success"]:
                            supabase_audio_url = audio_upload["url"]
                            print(f"☁️ Audio uploaded to Supabase")
                        
                        download_success = True
                        
                    except Exception as dl_err:
                        retry_count += 1
                        if audio_path and os.path.exists(audio_path):
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                        
                        if retry_count >= max_retries:
                            raise Exception(f"Download failed: {str(dl_err)[:100]}")
                        
                        print(f"⚠️ Retry {retry_count}: {str(dl_err)[:50]}")
                        import time
                        time.sleep(3 * retry_count)
                
                if download_success and audio_path and os.path.exists(audio_path):
                    transcription_success = False
                    transcription_retries = 3
                    transcription_attempt = 0
                    
                    while transcription_attempt < transcription_retries and not transcription_success:
                        try:
                            print(f"🎙️ Transcribing with Speaker Diarization (Attempt {transcription_attempt + 1}/{transcription_retries}): {episode_title[:40]}")
                            
                            config = aai.TranscriptionConfig(speaker_labels=True)
                            transcriber = aai.Transcriber()
                            transcript = transcriber.transcribe(audio_path, config=config)
                            
                            poll_count = 0
                            max_polls = 200
                            
                            while poll_count < max_polls:
                                try:
                                    if transcript.status == aai.TranscriptStatus.completed:
                                        transcription_success = True
                                        break
                                    elif transcript.status == aai.TranscriptStatus.error:
                                        raise Exception(f"Transcription API error: {transcript.error}")
                                    
                                    import time
                                    time.sleep(3)
                                    poll_count += 1
                                    transcript = transcriber.get_transcript(transcript.id)
                                    
                                except Exception as poll_err:
                                    print(f"⚠️ Poll error: {str(poll_err)[:50]}")
                                    raise
                            
                            if not transcription_success and poll_count >= max_polls:
                                raise Exception("Transcription timeout")
                            
                            if transcription_success and transcript.text:
                                # Get speaker count
                                speaker_count = 0
                                if hasattr(transcript, 'utterances') and transcript.utterances:
                                    speaker_count = len(set(u.speaker for u in transcript.utterances))
                                
                                print(f"👥 Detected {speaker_count} speakers")
                                
                                content = f"Podcast: {feed_info.get('title', 'Unknown')}\n"
                                content += f"Episode: {episode_title}\n"
                                content += f"Speakers Detected: {speaker_count}\n"
                                
                                if episode.get('description'):
                                    content += f"Description: {episode['description']}\n"
                                
                                # Try to format transcript with speaker labels, fall back to plain text
                                try:
                                    formatted_transcript = format_transcript_with_speakers(transcript)
                                    content += f"\nTranscript with Speaker Labels:\n{formatted_transcript}"
                                    print("✅ Transcript formatted with speaker labels")
                                except Exception as format_err:
                                    print(f"⚠️ Transcript formatting failed: {str(format_err)[:50]}")
                                    content += f"\nTranscript:\n{transcript.text}"
                                    print("📝 Using plain text transcript instead")
                                
                                pdf_filename = f"{safe_filename[:80]}_{uuid.uuid4().hex[:4]}"
                                
                                # Try PDF generation, but continue even if it fails
                                try:
                                    pdf_result = transcript_to_pdf_with_speakers(
                                        transcript,
                                        pdf_filename,
                                        upload_to_supabase=True
                                    )
                                    
                                    if pdf_result.get("supabase_uploaded"):
                                        supabase_pdf_url = pdf_result["supabase_url"]
                                        print(f"☁️ PDF uploaded to Supabase")
                                    elif pdf_result.get("error"):
                                        print(f"⚠️ PDF generation failed: {pdf_result['error']}")
                                        print("📝 Continuing without PDF - transcription still successful")
                                    else:
                                        print("⚠️ PDF generation skipped - continuing without PDF")
                                except Exception as pdf_err:
                                    print(f"⚠️ PDF generation error: {str(pdf_err)[:50]}")
                                    print("📝 Continuing without PDF - transcription still successful")
                                
                                splitter = get_dynamic_splitter(content)
                                chunks = splitter.create_documents([content], metadatas=[{
                                    "source": audio_url,
                                    "episode_title": episode_title,
                                    "feed_url": feed_url,
                                    "feed_title": feed_info.get('title', 'Unknown'),
                                    "content_type": "episode_transcript",
                                    "speaker_count": speaker_count,
                                    "supabase_audio_url": supabase_audio_url,
                                    "supabase_pdf_url": supabase_pdf_url
                                }])
                                
                                vs = load_vectorstore()
                                if vs is None:
                                    vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
                                else:
                                    vs.add_documents(chunks)
                                
                                print(f"✅ Stored: {episode_title[:40]} ({len(chunks)} chunks, {speaker_count} speakers)")
                                processing_status[task_id]["success"] += 1
                                episode_success = True
                                metadata_manager.add_episode(
                                    feed_url, 
                                    episode.get('guid', ''), 
                                    episode_title,
                                    audio_url=supabase_audio_url,
                                    pdf_url=supabase_pdf_url
                                )
                                break
                            
                        except Exception as transcribe_err:
                            transcription_attempt += 1
                            error_detail = str(transcribe_err)
                            
                            if "getaddrinfo failed" in error_detail or "Connection" in error_detail:
                                print(f"⚠️ Network error (Attempt {transcription_attempt}/{transcription_retries}): {error_detail[:100]}")
                                if transcription_attempt < transcription_retries:
                                    import time
                                    wait_time = 10 * transcription_attempt
                                    print(f"⏳ Waiting {wait_time}s before retry...")
                                    time.sleep(wait_time)
                                else:
                                    error_msg = f"Transcription failed after {transcription_retries} attempts: {episode_title} - Network error"
                                    processing_status[task_id]["errors"].append(error_msg)
                                    processing_status[task_id]["unsuccessful"] += 1
                                    print(f"❌ {error_msg}")
                            else:
                                error_msg = f"Transcription error: {episode_title} - {error_detail[:100]}"
                                processing_status[task_id]["errors"].append(error_msg)
                                processing_status[task_id]["unsuccessful"] += 1
                                print(f"❌ {error_msg}")
                                break
                    
                    # Clean up temp audio after processing
                    if audio_path and os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                    
                    if not transcription_success:
                        error_msg = f"Transcription timeout/failed: {episode_title}"
                        processing_status[task_id]["errors"].append(error_msg)
                        processing_status[task_id]["unsuccessful"] += 1
                        print(f"❌ {error_msg}")
                            
            elif not is_audio_url:
                # Non-audio URL - store metadata only
                print(f"📄 Metadata: {episode_title[:50]}")
                content = f"Episode: {episode_title}\nAudio URL: {audio_url}\nFeed: {feed_info.get('title', 'Unknown')}"
                if episode.get('description'):
                    content += f"\nDescription: {episode['description']}"
                
                splitter = get_dynamic_splitter(content)
                chunks = splitter.create_documents([content], metadatas=[{
                    "source": audio_url,
                    "episode_title": episode_title,
                    "feed_url": feed_url,
                    "feed_title": feed_info.get('title', 'Unknown'),
                    "content_type": "episode_metadata"
                }])
                
                vs = load_vectorstore()
                if vs is None:
                    vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
                else:
                    vs.add_documents(chunks)
                
                processing_status[task_id]["metadata"] += 1
                episode_success = True
                metadata_manager.add_episode(feed_url, episode.get('guid', ''), episode_title)
                
        except Exception as e:
            error_msg = f"Failed: {episode.get('title', 'Unknown')[:50]} - {str(e)[:100]}"
            processing_status[task_id]["errors"].append(error_msg)
            if not episode_success:
                processing_status[task_id]["unsuccessful"] += 1
            print(f"❌ {error_msg}")
        
        processed_count += 1
        processing_status[task_id]["current"] = processed_count
        
        if processed_count < len(episodes):
            import time
            time.sleep(3)
    
    processing_status[task_id]["status"] = "completed"
    print(f"🎉 Completed processing {processed_count} episodes")

# --- API ENDPOINTS ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    
    temp_dir = tempfile.mkdtemp()
    supabase_pdf_url = None
    
    try:
        tmp_path = os.path.join(temp_dir, filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        with open(tmp_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        if metadata_manager.is_file_duplicate(filename, file_hash):
            return {
                "success": False,
                "message": f"File '{filename}' already exists in the database.",
                "duplicate": True
            }

        content = ""

        if ext == "pdf":
            content = extract_text_from_pdf(tmp_path)
            
            # Upload PDF to Supabase
            pdf_upload = supabase_manager.upload_pdf(tmp_path, filename)
            if pdf_upload["success"]:
                supabase_pdf_url = pdf_upload["url"]

        elif ext in ["mp3", "wav", "m4a"]:
            # Upload audio to Supabase first
            audio_upload = supabase_manager.upload_audio(tmp_path, filename)
            supabase_audio_url = audio_upload["url"] if audio_upload["success"] else None
            
            # Transcribe with speaker diarization
            config = aai.TranscriptionConfig(speaker_labels=True)
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path, config=config)
            
            if transcript.status != aai.TranscriptStatus.error:
                # Try to format with speaker labels, fall back to plain text
                try:
                    content = format_transcript_with_speakers(transcript)
                    print("✅ Transcript formatted with speaker labels")
                except Exception as format_err:
                    print(f"⚠️ Transcript formatting failed: {str(format_err)[:50]}")
                    content = transcript.text
                    print("📝 Using plain text transcript instead")
                
                # Create and upload PDF to Supabase
                try:
                    pdf_result = transcript_to_pdf_with_speakers(transcript, filename, upload_to_supabase=True)
                    if pdf_result.get("supabase_uploaded"):
                        supabase_pdf_url = pdf_result["supabase_url"]
                    elif pdf_result.get("error"):
                        print(f"⚠️ PDF generation failed: {pdf_result['error']}")
                        print("📝 Continuing without PDF - transcription still successful")
                    else:
                        print("⚠️ PDF generation skipped - continuing without PDF")
                except Exception as pdf_err:
                    print(f"⚠️ PDF generation error: {str(pdf_err)[:50]}")
                    print("📝 Continuing without PDF - transcription still successful")
            else:
                content = ""

        elif ext == "txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

        if not content or not content.strip():
            return {"success": False, "message": f"Could not extract any text from {filename}."}

        splitter = get_dynamic_splitter(content)
        metadata = {
            "source": filename,
            "supabase_pdf_url": supabase_pdf_url
        }
        chunks = splitter.create_documents([content], metadatas=[metadata])
        
        vs = load_vectorstore()
        if vs is None:
            vs = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
        else:
            vs.add_documents(chunks)
        
        metadata_manager.add_file(filename, file_hash, len(chunks), supabase_pdf_url)

        return {
            "success": True, 
            "message": f"Successfully processed {filename} into {len(chunks)} chunks.",
            "chunks_count": len(chunks),
            "supabase_url": supabase_pdf_url
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/generate-content")
async def generate_content(req: ContentGenerationRequest):
    """Generate content based on input"""
    try:

        if req.generation_type == "episode_brief":
            result = content_generator.create_episode_brief(req.content)
        elif req.generation_type == "summary":
            result = content_generator.generate_summary(req.content)
        else:
            return {"success": False, "message": "Invalid generation type"}
        
        return result
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/enhanced-ingest-url")
async def enhanced_ingest_url(req: UrlRequest, background_tasks: BackgroundTasks):
    """Enhanced URL ingestion with deduplication, speaker diarization, and Supabase storage"""
    try:
        result = rss_processor.get_new_episodes(req.url)
        
        if result.get("error"):
            return {
                "success": False,
                "message": f"RSS Error: {result['error']}"
            }
        
        new_episodes = result["new_episodes"]
        is_new_feed = result["is_new_feed"]
        feed_info = result["feed_info"]
        
        if not new_episodes and not is_new_feed:
            return {
                "success": True,
                "message": "RSS feed already processed and no new episodes found.",
                "duplicate": True,
                "feed_info": feed_info,
                "already_processed": result.get("already_processed", 0)
            }
        
        if is_new_feed:
            metadata_manager.add_rss_feed(req.url, feed_info.get('title', 'Unknown Feed'))
        
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {
            "current": 0,
            "total": len(new_episodes),
            "status": "processing",
            "feed_url": req.url,
            "feed_info": feed_info,
            "is_new_feed": is_new_feed,
            "success": 0,
            "metadata": 0,
            "unsuccessful": 0,
            "errors": []
        }
        
        background_tasks.add_task(
            process_deduplicated_episodes_task, 
            req.url, new_episodes, feed_info, task_id
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Processing {len(new_episodes)} new episodes from {feed_info.get('title', 'Unknown Feed')} with speaker diarization and Supabase storage",
            "new_episodes_count": len(new_episodes),
            "is_new_feed": is_new_feed,
            "feed_info": feed_info,
            "total_episodes": result.get("total_episodes", 0),
            "already_processed": result.get("already_processed", 0)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/enhanced-ingest-multiple-urls")
async def enhanced_ingest_multiple_urls(req: MultiUrlRequest, background_tasks: BackgroundTasks):
    """Process multiple RSS feeds with deduplication, speaker diarization, and Supabase storage"""
    try:
        if not req.urls:
            return {"success": False, "message": "No URLs provided"}
        
        task_id = str(uuid.uuid4())
        
        processing_status[task_id] = {
            "current": 0,
            "total": len(req.urls),
            "status": "processing",
            "feeds_processed": 0,
            "results": [],
            "errors": []
        }
        
        background_tasks.add_task(
            process_multiple_feeds_task,
            req.urls,
            task_id
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Started processing {len(req.urls)} RSS feeds with speaker diarization and Supabase storage",
            "total_feeds": len(req.urls)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Endpoint for frontend to poll for progress"""
    return processing_status.get(task_id, {"current": 0, "total": 0, "status": "not_found"})

@app.post("/ingest-url")
async def ingest_url(req: UrlRequest, background_tasks: BackgroundTasks):
    try:
        file_links = get_all_file_links(req.url)
        if not file_links:
            return {"success": False, "message": "No links found."}

        task_id = str(uuid.uuid4())
        
        processing_status[task_id] = {
            "current": 0,
            "total": len(file_links), 
            "status": "processing",
            "success": 0,
            "unsuccessful": 0,
            "errors": []
        }

        background_tasks.add_task(process_in_batches_task, file_links, task_id)

        return {
            "success": True, 
            "task_id": task_id,
            "message": f"Started processing {len(file_links)} files with speaker diarization and Supabase storage."
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        history = []
        for m in request.history:
            if m.get("role") == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant":
                history.append(AIMessage(content=m["content"]))

        inputs = {
            "question": request.message,
            "messages": history,
            "answer": "",
            "context": "",
            "sources": [],
            "episode_titles": []
        }
        
        result = app_agent.invoke(inputs)
        
        # Debug: Print what we got from agents
        print(f"--- DEBUG: Result keys: {list(result.keys())} ---")
        if "sources" in result:
            print(f"--- DEBUG: Sources count: {len(result['sources'])} ---")
        if "episode_titles" in result:
            print(f"--- DEBUG: Episode titles: {result['episode_titles']} ---")
        
        response_data = {"answer": result["answer"]}
        
        # Add source information if available
        if "sources" in result and result["sources"]:
            response_data["sources"] = result["sources"]
            print(f"--- DEBUG: Added {len(result['sources'])} sources to response ---")
        
        if "episode_titles" in result and result["episode_titles"]:
            response_data["episode_titles"] = result["episode_titles"]
            print(f"--- DEBUG: Added {len(result['episode_titles'])} episode titles to response ---")
        
        # Final debug
        print(f"--- DEBUG: Final response keys: {list(response_data.keys())} ---")
            
        return response_data
    except Exception as e:
        print(f"Backend Error: {e}")
        return {"answer": f"A technical error occurred: {str(e)}"}

@app.delete("/clear")
async def clear_db():
    try:
        index = get_pinecone_index()
        if index:
            index.delete(delete_all=True)
            print("Cleared all vectors from Pinecone index")
        
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
            
        return {"success": True}
    except Exception as e:
        print(f"Error clearing database: {e}")
        return {"success": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )