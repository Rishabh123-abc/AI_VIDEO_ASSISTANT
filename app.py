import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from main import run_pipeline, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- SESSION STATE --------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False


# -------------------- CUSTOM UI --------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎬 AI Video Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Transcribe, summarize, extract insights, and chat with your content.'
    '</div>',
    unsafe_allow_html=True,
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("⚙️ Settings")

    language = st.selectbox(
        "Transcription language",
        options=["english", "hindi", "hinglish"],
        index=0,
    )

    st.divider()

    st.markdown("### Features")
    st.markdown(
        """
        - 🎙️ Audio/video transcription
        - 📝 AI-generated summary
        - ✅ Action item extraction
        - 💡 Key decision extraction
        - ❓ Open question extraction
        - 💬 Chat with your transcript
        - 📥 Download chat history
        """
    )

    if st.button("🗑️ Start New Session", use_container_width=True):
        st.session_state.result = None
        st.session_state.chat_history = []
        st.rerun()


# -------------------- INPUT SECTION --------------------

st.subheader("1. Upload or provide a video/audio source")

input_mode = st.radio(
    "Choose your input",
    ["Upload File", "YouTube / Direct URL"],
    horizontal=True,
)

uploaded_file = None
source_url = ""

if input_mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=[
            "mp3", "wav", "m4a", "ogg", "flac",
            "mp4", "mkv", "mov", "avi", "webm",
        ],
        help="Upload a supported audio or video file.",
    )
else:
    source_url = st.text_input(
        "Video/audio URL",
        placeholder="Paste a YouTube or supported media URL...",
    )

process_clicked = st.button(
    "🚀 Process Content",
    type="primary",
    use_container_width=True,
)

if process_clicked:
    source = None
    temp_path = None

    if input_mode == "Upload File":
        if uploaded_file is None:
            st.warning("Please upload an audio or video file first.")
        else:
            try:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                source = temp_path

            except Exception as exc:
                st.error(f"Could not prepare the uploaded file: {exc}")

    else:
        if not source_url.strip():
            st.warning("Please enter a URL first.")
        else:
            source = source_url.strip()

    if source:
        try:
            with st.spinner(
                "Processing content... This may take a few minutes."
            ):
                result = run_pipeline(source, language)

            st.session_state.result = result
            st.session_state.chat_history = []

            st.success("Content processed successfully!")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# -------------------- RESULTS --------------------

result = st.session_state.result

if result:
    st.divider()

    st.subheader("2. Processing Results")

    st.markdown(f"## 🎞️ {result.get('title', 'Untitled Content')}")

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions = (
        st.tabs(
            [
                "📝 Summary",
                "📄 Transcript",
                "✅ Action Items",
                "💡 Key Decisions",
                "❓ Open Questions",
            ]
        )
    )

    with tab_summary:
        st.markdown("### Summary")
        st.write(result.get("summary", "No summary available."))

        st.download_button(
            "📥 Download Summary",
            data=str(result.get("summary", "")),
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary",
        )

    with tab_transcript:
        transcript = str(result.get("transcript", ""))

        st.text_area(
            "Full Transcript",
            value=transcript,
            height=400,
            disabled=True,
        )

        st.download_button(
            "📥 Download Transcript",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain",
            key="download_transcript",
        )

    with tab_actions:
        st.write(result.get("action_items", "No action items available."))

    with tab_decisions:
        st.write(result.get("key_decisions", "No key decisions available."))

    with tab_questions:
        st.write(result.get("questions", "No open questions available."))

    st.divider()

    # -------------------- CHAT --------------------

    st.subheader("3. Chat with Your Content")
    st.caption(
        "Ask questions about the transcript. "
        "Your conversation stays available during this session."
    )

    # Display previous messages.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User enters a new question.
    user_question = st.chat_input(
        "Ask something about your video or audio..."
    )

    if user_question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(
                        result["rag_chain"],
                        user_question,
                    )

                answer = str(answer)
                st.markdown(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:
            error_message = f"Sorry, I couldn't answer that question: {exc}"

            with st.chat_message("assistant"):
                st.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

        st.rerun()

    # -------------------- DOWNLOAD CHAT --------------------

    st.divider()
    st.subheader("📥 Save Your Conversation")

    if st.session_state.chat_history:
        chat_lines = [
            f"AI Video Assistant Chat",
            f"Content: {result.get('title', 'Untitled Content')}",
            "=" * 60,
            "",
        ]

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                speaker = "YOU"
            else:
                speaker = "AI ASSISTANT"

            chat_lines.append(f"{speaker}:\n{message['content']}")
            chat_lines.append("")
            chat_lines.append("-" * 60)
            chat_lines.append("")

        chat_text = "\n".join(chat_lines)

        st.download_button(
            label="⬇️ Download Full Chat (.txt)",
            data=chat_text,
            file_name="ai_video_assistant_chat.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True,
            key="download_full_chat",
        )

        st.caption(
            f"{len(st.session_state.chat_history)} messages ready to download."
        )

    else:
        st.info("Start chatting to enable the full chat download.")

else:
    st.info(
        "👆 Upload a file or enter a URL, then click "
        "'Process Content' to get started."
    )