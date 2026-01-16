import streamlit as st
import requests
import time

# --- CONFIGURATION ---
BACKEND_URL = "http://127.0.0.1:8000"

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
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "chat"

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
    else:
        st.markdown("### 📚 Knowledge Base")
        st.markdown("Upload documents and RSS feeds to build your knowledge base.")

# --- MAIN CONTENT AREA ---
if st.session_state.current_tab == "chat":
    # CHAT INTERFACE
    st.markdown("## 💬 Chat with Your Knowledge Base")
    st.markdown("---")
    
    # Display chat history
    for m in st.session_state.ui_messages:
        with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
    
    # Chat input at bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.ui_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Thinking...")
            
            try:
                payload = {
                    "message": prompt,
                    "history": st.session_state.ui_messages[:1]
                }
                
                res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
                
                if res.status_code == 200:
                    answer = res.json().get("answer", "No response received.")
                    message_placeholder.markdown(answer)
                    st.session_state.ui_messages.append({"role": "assistant", "content": answer})
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
                            
                    elif data.get("duplicate"):
                        st.warning("⚠️ " + data.get('message', 'File already exists'))
                    else:
                        st.error("❌ " + data.get("message", "Upload failed."))
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")
        
            
    # TAB 2: RSS FEED
    with upload_tab2:
        st.markdown("### Add RSS Feeds")
        
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
                                    # Live counters — UPDATED METRICS
                                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)  # ← 4 columns now
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
                                            metadata_n = status_data.get("metadata", 0)  # ← NEW
                                            failed_n = status_data.get("unsuccessful", 0)

                                            transcribed_box.metric("✅ Transcribed", success_n)
                                            metadata_box.metric("📄 Metadata", metadata_n)
                                            failed_box.metric("❌ Failed", failed_n)
                                            total_box.metric("📦 Total", current)
                                            
                                            if total > 0:
                                                progress_val = min(current / total, 1.0)
                                                progress_bar.progress(progress_val)
                                                status_text.text(f"Processing episode {current} of {total}...")
                                            
                                            if state == "completed":
                                                is_done = True
                                                status_text.success(f"🎊 Completed! {success_n} transcribed, {metadata_n} metadata, {failed_n} failed (total {current}).")
                                                
                                        except Exception:
                                            pass
                                        
                                        time.sleep(2)
                        else:
                            # Multiple feeds processing (unchanged)
                            st.success(f"✅ Processing {len(urls_to_process)} feeds")
                            
                            if task_id:
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
                                                        st.write(f"   - New Episodes: {result.get('new_episodes', 0)}")
                                                    else:
                                                        st.error(f"❌ Failed: {result.get('feed_url', '')}")
                                        
                                        if state == "completed":
                                            is_done = True
                                            successful = sum(1 for r in results if r.get("success"))
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("✅ Success", successful)
                                            with col2:
                                                st.metric("❌ Unsuccessful", len(results) - successful)
                                            with col3:
                                                total_episodes = sum(r.get('new_episodes', 0) for r in results if r.get("success"))
                                                st.metric("📚 Episodes", total_episodes)
                                            
                                            status_text.success("🎊 All feeds processed!")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
                                    
                                    time.sleep(2)
                    else:
                        st.error(data.get("message", "Processing failed"))
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")