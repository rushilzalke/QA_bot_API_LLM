import streamlit as st
import os
from bot_logic import process_pdf, get_answer

st.set_page_config(page_title="PDF Q&A Bot", page_icon=":books:")
st.title("PDF Q&A Bot :books 📄:")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing / Analyzing PDF..."):
            st.session_state.vector_store = process_pdf("temp.pdf")
        st.success("PDF processed successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
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
    
    if "vector_store" in st.session_state:
        with st.chat_message("Assistant"):
            response = get_answer(st.session_state.vector_store, prompt)
            st.markdown(f"**Bot:** {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("Assistant"):
            st.markdown("**Bot:** Please upload and process a PDF first.")