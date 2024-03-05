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
    if 'api_key' not in st.session_state:
        api_key_input.header("0. Input API keys")
        user_input = api_key_input.text_input('Enter your text here')
        if user_input:
            os.environ["OPENAI_API_KEY"] = user_input
            openai.api_key = os.environ['OPENAI_API_KEY']
            llm3 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
            llm4 = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
            st.session_state['api_key'] = user_input
            api_key_input.success(f"✅ You API: **{user_input}**")
            api_key_input.markdown("---")

    # 1. Upload file
    if 'api_key' in st.session_state:
        upload_file_placeholder.header("1. Upload PDF file")
        uploaded_file = upload_file_placeholder.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", type=['pdf'], accept_multiple_files=False)

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf.flush()
                temp_pdf_path = temp_pdf.name
                st.session_state['temp_pdf_path'] = temp_pdf_path
                upload_file_placeholder.success(f"✅ You selected: **{temp_pdf_path}**")
                upload_file_placeholder.markdown("---")

    # 2. Select model
    if 'temp_pdf_path' in st.session_state:
        model_choice_placeholder.header("2. Select model")
        model_choice = model_choice_placeholder.radio(
            "*Note: GPT-4 will charge more credits than GPT-3 (typically 4 times more compared to GPT-3).*",
            ('gpt-3.5-turbo-1106', 'gpt-4-1106-preview')
        )
        if model_choice:
            st.session_state['model_choice'] = model_choice
            model_choice_placeholder.success(f"✅ You selected: **{model_choice}**")
            model_choice_placeholder.markdown("---")

    # 3. Select chain type
    if 'model_choice' in st.session_state:
        chain_choice_placeholder.header("3. Select chain type")
        chain_choice = chain_choice_placeholder.radio(
            "*Note: refine will charge more credits map-reduce.*",
            ('map-reduce', 'refine')
        )
        if chain_choice:
            st.session_state['chain_choice'] = chain_choice
            chain_choice_placeholder.success(f"✅ You selected: **{chain_choice}**")
            chain_choice_placeholder.markdown("---")

    # Button with function call
    if 'chain_choice' in st.session_state:
        if confirm_button_placeholder.button("Confirm Selections"):
            # Display a message to the user
            st.text("Processing your selections... Please wait.")
            
            # Initialize a progress bar
            progress_bar = st.progress(0)

            # Assuming the processing takes a certain number of steps, update the progress
            for percent_complete in range(100):
                time.sleep(0.1) # Simulate processing delay
                progress_bar.progress(percent_complete + 1)

            # Your existing code here for processing
            docs = pdf_loader(st.session_state['temp_pdf_path'])

            if (st.session_state['model_choice'] == 'gpt-3.5-turbo-1106') & (st.session_state['chain_choice'] == 'map-reduce'):
                sum = map_reduce_result(docs, llm3)
            elif (st.session_state['model_choice'] == 'gpt-3.5-turbo-1106') & (st.session_state['chain_choice'] == 'refine'):
                sum = refine_result(docs, llm3)
            elif (st.session_state['model_choice'] == 'gpt-4-1106-preview') & (st.session_state['chain_choice'] == 'map-reduce'):
                sum = map_reduce_result(docs, llm4)
            elif (st.session_state['model_choice'] == 'gpt-4-1106-preview') & (st.session_state['chain_choice'] == 'refine'):
                sum = refine_result(docs, llm4)

            # After processing is complete, show the success message
            st.success(sum)