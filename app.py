import streamlit as st

st.set_page_config(page_title="PDF Q&A Bot", page_icon=":books:")
st.title("PDF Q&A Bot 📄")

with st.sidebar:
    # ── LLM Selector ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 Choose LLM Backend")

    llm_choice = st.radio(
        label="LLM Backend",
        options=["Google Gemini API", "HuggingFace (Llama-3.1)"],
        index=0,
        label_visibility="collapsed",
    )

    # Show confirm button — only activates the switch when clicked
    confirm = st.button("✅ Confirm Model", use_container_width=True)

    if confirm:
        # Only re-process if the model actually changed
        if st.session_state.get("active_llm") != llm_choice:
            st.session_state.active_llm    = llm_choice
            st.session_state.messages      = []       # clear chat on model switch
            st.session_state.vector_store  = None     # force re-embed with new backend
            st.session_state.pop("uploaded_filename", None)
            st.rerun()

    # Show which model is currently active
    active = st.session_state.get("active_llm", "Google Gemini API")
    if active == "Google Gemini API":
        st.caption("🟢 Active: Google Gemini API")
    else:
        st.caption("🟡 Active: HuggingFace (Llama-3.1)")

    st.divider()

    # ── PDF Uploader ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF file", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_changed = st.session_state.get("uploaded_filename") != uploaded_file.name

        if file_changed or st.session_state.get("vector_store") is None:
            # Import the correct backend based on confirmed active model
            active_llm = st.session_state.get("active_llm", "Google Gemini API")
            if active_llm == "Google Gemini API":
                from bot_logic import process_pdf, get_answer
            else:
                from bot_logic_hf import process_pdf, get_answer

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing / Analyzing PDF..."):
                try:
                    st.session_state.vector_store      = process_pdf("temp.pdf")
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.messages          = []
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
        else:
            st.success(f"✅ {uploaded_file.name} ready")

# ── Load correct backend for answering ───────────────────────────────────────
active_llm = st.session_state.get("active_llm", "Google Gemini API")
if active_llm == "Google Gemini API":
    from bot_logic import process_pdf, get_answer
else:
    from bot_logic_hf import process_pdf, get_answer

# ── Chat Interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# User input
if prompt := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("User"):
        st.markdown(f"**You:** {prompt}")

    if st.session_state.get("vector_store"):
        with st.chat_message("Assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(st.session_state.vector_store, prompt)
            st.markdown(f"**Bot:** {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("Assistant"):
            st.markdown("**Bot:** Please upload and process a PDF first.")