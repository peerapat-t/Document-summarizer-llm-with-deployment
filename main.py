import os
import openai
from langchain.chat_models import ChatOpenAI

from source_loader import pdf_loader, word_loader, powerpoint_loader, web_loader, youtube_loader, text_loader
from summarizer import map_reduce_result, refine_result

import streamlit as st
import tempfile

import time


# Sidebar title and color customization
st.sidebar.title("Configuration")
st.sidebar.markdown("<style>.sidebar .sidebar-content { background-color: #212529; color: white; }</style>", unsafe_allow_html=True,)

# Create tabs with emojis
tab1, tab2, tab3 = st.tabs(["📝 PDF", "📩 Word", "🌍 Power Point"])

with tab1:

    # Create placeholders for the widgets
    api_key_input = st.empty()
    upload_file_placeholder = st.empty()
    model_choice_placeholder = st.empty()
    chain_choice_placeholder = st.empty()
    confirm_button_placeholder = st.empty()

    # 0. Set API
    st.header("0. Input API keys")
    user_input = st.text_input('Enter your text here')

    if user_input:
        # Only set the API key and initialize models if the user has entered their API key
        os.environ["OPENAI_API_KEY"] = user_input
        openai.api_key = os.environ['OPENAI_API_KEY']
        llm3 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
        llm4 = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")

        st.success(f"✅ You API: **{user_input}**")
    else:
        # If the user has not entered their API key, display a message asking them to do so
        st.warning("Please enter your API key to proceed.")

    st.markdown("---")

    # 1. Upload file
    st.header("1. Upload PDF file")
    uploaded_file = st.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", type=['pdf'], accept_multiple_files=False)
    if uploaded_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf.flush()
            temp_pdf_path = temp_pdf.name
        st.success(f"✅ You selected: **{temp_pdf_path}**")
    st.markdown("---")

    # 2. Select model
    st.header("2. Select model")
    model_choice = st.radio(
        "*Note: GPT-4 will charge more credits than GPT-3 (typically 4 times more compared to GPT-3).*", # Removed the label here
        ('gpt-3.5-turbo-1106', 'gpt-4-1106-preview')
    )
    if model_choice:
        st.success(f"✅ You selected: **{model_choice}**")
    st.markdown("---")

    # 3. Select chain type
    st.header("3. Select chain type")
    chain_choice = st.radio(
        "*Note: refine will charge more credits map-reduce.*", # Removed the label here
        ('map-reduce', 'refine')
    )
    if chain_choice:
        st.success(f"✅ You selected: **{chain_choice}**")
    st.markdown("---")

    # Button with function call
    if st.button("Confirm Selections"):
        # Display a message to the user
        st.text("Processing your selections... Please wait.")
        
        # Initialize a progress bar
        progress_bar = st.progress(0)

        # Assuming the processing takes a certain number of steps, update the progress
        for percent_complete in range(100):
            time.sleep(0.1)  # Simulate processing delay
            progress_bar.progress(percent_complete + 1)

        # Your existing code here for processing
        docs = pdf_loader(temp_pdf_path)

        if (model_choice == 'gpt-3.5-turbo-1106') & (chain_choice == 'map-reduce'):
            sum = map_reduce_result(docs, llm3)
        elif (model_choice == 'gpt-3.5-turbo-1106') & (chain_choice == 'refine'):
            sum = refine_result(docs, llm3)
        elif (model_choice == 'gpt-4-1106-preview') & (chain_choice == 'map-reduce'):
            sum = map_reduce_result(docs, llm4)
        elif (model_choice == 'gpt-4-1106-preview') & (chain_choice == 'refine'):
            sum = refine_result(docs, llm4)

        # After processing is complete, show the success message
        st.success(sum)