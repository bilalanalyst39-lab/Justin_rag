import streamlit as st
import requests
import os
from typing import List, Dict
import time

# --- CONFIGURATION ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG Knowledge Base System",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #0d8b6f;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        border: 2px dashed #d1d5db;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
    }
    
    /* Headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #10a37f;
    }
    
    /* Tabs in main area */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f3f4f6;
    }
    
    /* Upload sections */
    .upload-box {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    /* Source citation styling */
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #10a37f;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 6px;
    }
    
    .source-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    .source-link {
        color: #10a37f;
        text-decoration: none;
        font-size: 14px;
        margin-right: 12px;
    }
    
    .source-link:hover {
        text-decoration: underline;
    }
    
    .episode-badge {
        display: inline-block;
        background-color: #10a37f;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        margin: 4px;
        font-weight: 500;
    }
    
    /* Supabase link styling */
    .supabase-link-box {
        background-color: #e0f2fe;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 12px;
        margin: 12px 0;
    }
    
    .supabase-link-box a {
        color: #0284c7;
        font-weight: 500;
        text-decoration: none;
    }
    
    .supabase-link-box a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "chat"

# --- HELPER FUNCTIONS ---
def display_sources(sources, episode_titles):
    """Display source citations with Supabase links"""
    if episode_titles:
        st.markdown("---")
        st.markdown("**📚 Referenced Episodes:**")
        for title in episode_titles[:5]:  # Show top 5 episodes
            st.markdown(f'<span class="episode-badge">🎙️ {title}</span>', unsafe_allow_html=True)
    
    if sources:
        st.markdown("---")
        with st.expander("📖 View Sources & Downloads", expanded=False):
            for idx, source in enumerate(sources[:5], 1):  # Show top 5 sources
                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                
                episode_title = source.get('episode_title', 'Unknown')
                feed_title = source.get('feed_title', '')
                content_type = source.get('content_type', '')
                
                st.markdown(f'<div class="source-title">📄 Source {idx}: {episode_title}</div>', unsafe_allow_html=True)
                
                if feed_title:
                    st.markdown(f"**Podcast:** {feed_title}")
                
                if content_type:
                    st.markdown(f"**Type:** {content_type}")
                
                # Display Supabase links
                col1, col2 = st.columns(2)
                
                with col1:
                    pdf_url = source.get('supabase_pdf_url')
                    if pdf_url:
                        st.markdown(f'<a href="{pdf_url}" target="_blank" class="source-link">📥 Download Transcript PDF</a>', unsafe_allow_html=True)
                
                with col2:
                    audio_url = source.get('supabase_audio_url')
                    if audio_url:
                        st.markdown(f'<a href="{audio_url}" target="_blank" class="source-link">🎵 Download Audio</a>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")

def display_chat_message(role, content, sources=None, episode_titles=None):
    """Display chat message with optional sources"""
    with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
        st.markdown(content)
        
        # Display sources if available (only for assistant messages)
        if role == "assistant" and (sources or episode_titles):
            display_sources(sources or [], episode_titles or [])

# --- SIDEBAR (Only Navigation) ---
with st.sidebar:
    st.markdown("# 🤖 RAG System")
    st.markdown("---")
    
    # Navigation
    if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.current_tab == "chat" else "secondary"):
        st.session_state.current_tab = "chat"
        st.rerun()
    
    if st.button("📚 Upload Knowledge", use_container_width=True, type="primary" if st.session_state.current_tab == "knowledge" else "secondary"):
        st.session_state.current_tab = "knowledge"
        st.rerun()
    
    st.markdown("---")
    
    # Additional info
    if st.session_state.current_tab == "chat":
        st.markdown("### 💬 Chat Mode")
        st.markdown("Ask questions about your uploaded documents.")
        st.markdown("")
        st.markdown("**Features:**")
        st.markdown("- 🎙️ Speaker-aware responses")
        st.markdown("- 📚 Episode citations")
        st.markdown("- 📥 Download sources")
    else:
        st.markdown("### 📚 Knowledge Base")
        st.markdown("Upload documents and RSS feeds to build your knowledge base.")
        st.markdown("")
        st.markdown("**Supported:**")
        st.markdown("- 📄 PDF documents")
        st.markdown("- 📝 Text files")
        st.markdown("- 🎙️ Podcast RSS feeds")
        st.markdown("- 👥 Speaker diarization")

# --- MAIN CONTENT AREA ---
if st.session_state.current_tab == "chat":
    # CHAT INTERFACE
    st.markdown("## 💬 Chat with Your Knowledge Base")
    st.markdown("---")
    
    # Display chat history
    for m in st.session_state.ui_messages:
        display_chat_message(
            m["role"], 
            m["content"],
            sources=m.get("sources"),
            episode_titles=m.get("episode_titles")
        )
    
    # Chat input at bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.ui_messages.append(user_message)
        
        display_chat_message("user", prompt)
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Thinking...")
            
            try:
                payload = {
                    "message": prompt,
                    "history": st.session_state.ui_messages[:-1]  # Exclude current message
                }
                
                res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
                
                if res.status_code == 200:
                    response_data = res.json()
                    answer = response_data.get("answer", "No response received.")
                    sources = response_data.get("sources", [])
                    episode_titles = response_data.get("episode_titles", [])
                    
                    # Clear thinking message
                    message_placeholder.empty()
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources or episode_titles:
                        display_sources(sources, episode_titles)
                    
                    # Save to history with sources
                    assistant_message = {
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "episode_titles": episode_titles
                    }
                    st.session_state.ui_messages.append(assistant_message)
                else:
                    error_msg = "❌ Backend error occurred. Please try again."
                    message_placeholder.markdown(error_msg)
                    st.session_state.ui_messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timeout. Please try again."
                message_placeholder.markdown(error_msg)
                st.session_state.ui_messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"❌ Connection error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.ui_messages.append({"role": "assistant", "content": error_msg})

else:
    # UPLOAD KNOWLEDGE INTERFACE
    st.markdown("## 📚 Upload Knowledge")
    st.markdown("Add documents and RSS feeds to build your knowledge base")
    st.markdown("---")
    
    # Tabs for different upload types
    upload_tab1, upload_tab2 = st.tabs(["📄 PDF Upload", "📡 RSS Feed"])
    
    # TAB 1: PDF UPLOAD
    with upload_tab1:
        st.markdown("### Upload PDF Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF or text file",
                type=["pdf", "txt"],
                help="Upload PDF documents or text files to add to your knowledge base"
            )
        
        with col2:
            st.markdown("")
            st.markdown("")
            process_btn = st.button("🚀 Process Document", use_container_width=True, disabled=uploaded_file is None)
        
        if process_btn and uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    
                    data = res.json()
                    if res.status_code == 200 and data.get("success"):
                        st.success("✅ " + data.get("message", "File processed!"))
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("📄 File", uploaded_file.name)
                        with col_b:
                            st.metric("📊 Chunks", data.get('chunks_count', 'N/A'))
                        with col_c:
                            st.metric("✅ Status", "Processed")
                        
                        # Display Supabase URL if available
                        supabase_url = data.get("supabase_url")
                        if supabase_url:
                            st.markdown("---")
                            st.markdown("**☁️ Cloud Storage:**")
                            st.markdown(
                                f'<div class="supabase-link-box">'
                                f'📥 <a href="{supabase_url}" target="_blank">Download from Cloud Storage</a>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.info("💡 Your document is securely stored in the cloud and ready to be queried!")
                            
                    elif data.get("duplicate"):
                        st.warning("⚠️ " + data.get('message', 'File already exists'))
                    else:
                        st.error("❌ " + data.get("message", "Upload failed."))
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")
        
            
    # TAB 2: RSS FEED
    with upload_tab2:
        st.markdown("### Add RSS Feeds")
        st.info("🎙️ Podcast episodes will be transcribed with **speaker diarization** and stored in the cloud")
        
        # Feed mode selection
        feed_mode = st.radio(
            "Select feed mode:",
            ["Single Feed", "Multiple Feeds"],
            horizontal=True
        )
        
        st.markdown("")
        
        if feed_mode == "Single Feed":
            # Single feed input
            col1, col2 = st.columns([3, 1])
            
            with col1:
                input_url = st.text_input(
                    "RSS Feed URL",
                    placeholder="https://example.com/podcast/rss",
                    help="Enter a single RSS feed URL"
                )
            
            with col2:
                st.markdown("")
                st.markdown("")
                process_rss = st.button("🚀 Process Feed", use_container_width=True, disabled=not input_url)
            
            urls_to_process = [input_url] if input_url else []
            
        else:
            # Multiple feeds input
            input_urls = st.text_area(
                "RSS Feed URLs (one per line)",
                placeholder="https://feed1.com/rss\nhttps://feed2.com/rss\nhttps://feed3.com/rss",
                height=150,
                help="Enter multiple RSS feed URLs, one per line"
            )
            
            urls_to_process = [url.strip() for url in input_urls.split('\n') if url.strip()] if input_urls else []
            
            if urls_to_process:
                st.info(f"📊 {len(urls_to_process)} feed(s) ready to process")
            
            process_rss = st.button("🚀 Process All Feeds", use_container_width=True, disabled=not urls_to_process)
        
        # Process RSS feeds
        if process_rss and urls_to_process:
            with st.spinner("Processing RSS feeds..."):
                try:
                    if len(urls_to_process) == 1:
                        res = requests.post(
                            f"{BACKEND_URL}/enhanced-ingest-url",
                            json={"url": urls_to_process[0]},
                            timeout=30
                        )
                    else:
                        res = requests.post(
                            f"{BACKEND_URL}/enhanced-ingest-multiple-urls",
                            json={"urls": urls_to_process},
                            timeout=30
                        )
                    
                    data = res.json()
                    
                    if data.get("success"):
                        task_id = data.get("task_id")
                        
                        if len(urls_to_process) == 1:
                            # Single feed processing
                            feed_info = data.get("feed_info", {})
                            
                            if data.get("duplicate"):
                                st.warning(f"⚠️ Feed already processed")
                                st.info(f"📊 Already processed: {data.get('already_processed', 0)} episodes")
                            else:
                                st.success(f"✅ Processing: {feed_info.get('title', 'Unknown Feed')}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🆕 New Episodes", data.get("new_episodes_count", 0))
                                with col2:
                                    st.metric("📚 Total Episodes", data.get("total_episodes", 0))
                                with col3:
                                    st.metric("📋 Already Done", data.get("already_processed", 0))
                                
                                if task_id:
                                    st.markdown("---")
                                    st.markdown("### 🔄 Processing Progress")
                                    st.info("🎙️ Transcribing with speaker diarization & uploading to cloud storage...")
                                    
                                    # Live counters with 4 metrics
                                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                                    transcribed_box = mcol1.empty()
                                    metadata_box = mcol2.empty()
                                    failed_box = mcol3.empty()
                                    total_box = mcol4.empty()

                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    is_done = False
                                    while not is_done:
                                        try:
                                            status_res = requests.get(f"{BACKEND_URL}/status/{task_id}")
                                            status_data = status_res.json()
                                            
                                            current = status_data.get("current", 0)
                                            total = status_data.get("total", 0)
                                            state = status_data.get("status", "")

                                            success_n = status_data.get("success", 0)
                                            metadata_n = status_data.get("metadata", 0)
                                            failed_n = status_data.get("unsuccessful", 0)

                                            transcribed_box.metric("✅ Transcribed", success_n)
                                            metadata_box.metric("📄 Metadata Only", metadata_n)
                                            failed_box.metric("❌ Failed", failed_n)
                                            total_box.metric("📦 Processed", current)
                                            
                                            if total > 0:
                                                progress_val = min(current / total, 1.0)
                                                progress_bar.progress(progress_val)
                                                status_text.text(f"Processing episode {current} of {total}...")
                                            
                                            if state == "completed":
                                                is_done = True
                                                progress_bar.progress(1.0)
                                                status_text.success(
                                                    f"🎊 Completed! "
                                                    f"✅ {success_n} transcribed with speakers | "
                                                    f"📄 {metadata_n} metadata only | "
                                                    f"❌ {failed_n} failed"
                                                )
                                                
                                                st.balloons()
                                                
                                                st.markdown("---")
                                                st.success("✨ All episodes are now in your knowledge base and ready to query!")
                                                
                                        except Exception as e:
                                            st.error(f"Status check error: {e}")
                                        
                                        time.sleep(2)
                        else:
                            # Multiple feeds processing
                            st.success(f"✅ Processing {len(urls_to_process)} feeds")
                            
                            if task_id:
                                st.markdown("---")
                                st.markdown("### 🔄 Processing Progress")
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                results_expander = st.expander("📊 Processing Results", expanded=True)
                                
                                is_done = False
                                while not is_done:
                                    try:
                                        status_res = requests.get(f"{BACKEND_URL}/status/{task_id}")
                                        status_data = status_res.json()
                                        
                                        current = status_data.get("current", 0)
                                        total = status_data.get("total", 0)
                                        state = status_data.get("status", "")
                                        current_feed = status_data.get("current_feed", "")
                                        results = status_data.get("results", [])
                                        
                                        if total > 0:
                                            progress_val = min(current / total, 1.0)
                                            progress_bar.progress(progress_val)
                                            status_text.text(f"Processing feed {current} of {total}: {current_feed}")
                                        
                                        # Show results
                                        if results:
                                            with results_expander:
                                                for result in results:
                                                    if result.get("success"):
                                                        st.success(f"✅ {result.get('feed_title', 'Unknown')}")
                                                        col1, col2 = st.columns(2)
                                                        with col1:
                                                            st.write(f"   📚 New Episodes: {result.get('new_episodes', 0)}")
                                                        with col2:
                                                            st.write(f"   ✨ Processed: {result.get('episodes_processed', 0)}")
                                                    else:
                                                        st.error(f"❌ Failed: {result.get('feed_url', '')}")
                                        
                                        if state == "completed":
                                            is_done = True
                                            successful = sum(1 for r in results if r.get("success"))
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("✅ Successful Feeds", successful)
                                            with col2:
                                                st.metric("❌ Failed Feeds", len(results) - successful)
                                            with col3:
                                                total_episodes = sum(r.get('new_episodes', 0) for r in results if r.get("success"))
                                                st.metric("📚 Total Episodes", total_episodes)
                                            
                                            status_text.success("🎊 All feeds processed!")
                                            st.balloons()
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
                                    
                                    time.sleep(2)
                    else:
                        st.error(data.get("message", "Processing failed"))
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# --- FOOTER ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🤖 **RAG Knowledge Base System**")
with col2:
    st.markdown("☁️ **Powered by Supabase Storage**")
with col3:
    st.markdown("🎙️ **Speaker Diarization Enabled**")