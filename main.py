import os
import openai
from langchain.chat_models import ChatOpenAI

from source_loader import pdf_loader, word_loader, powerpoint_loader, web_loader, youtube_loader, text_loader
from summarizer import map_reduce_result, refine_result, translate_to_thai

import streamlit as st
import tempfile

import time


# Sidebar title and color customization
st.sidebar.title("Configuration")
st.sidebar.markdown("<style>.sidebar .sidebar-content { background-color: #212529; color: white; }</style>", unsafe_allow_html=True,)

# Create tabs with emojis
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📚 PDF", "📃 Word", "📊 Powerpoint", "🌍 Website", "📺 Youtube", "✍️ Just text"])
with tab1:

    # 1. Set API
    st.header("1. Input API keys")
    api_key_input = st.empty()
    api_key_input = st.text_input('Enter your text here')
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        openai.api_key = os.environ['OPENAI_API_KEY']
        llm3 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
        llm4 = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
        st.success(f"✅ You API: **{api_key_input}**")
    else:
        st.warning("Please enter your API key to proceed.")
    st.markdown("---")

    # 2. Upload file
    st.header("2. Upload PDF file")
    uploaded_file_input = st.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", type=['pdf'], accept_multiple_files=False)
    if uploaded_file_input:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file_input.read())
            temp_pdf.flush()
            temp_pdf_path = temp_pdf.name
        st.success(f"✅ You selected: **{temp_pdf_path}**")
    st.markdown("---")

    # 3. Select model
    st.header("3. Select model")
    model_choice_input = st.radio(
        "*Note: GPT-4 will charge more credits than GPT-3 (typically 4 times more compared to GPT-3).*",
        ('gpt-3.5-turbo-1106', 'gpt-4-1106-preview')
    )
    if model_choice_input:
        st.success(f"✅ You selected: **{model_choice_input}**")
    st.markdown("---")

    # 4. Select chain type
    st.header("4. Select chain type")
    chain_choice_input = st.radio(
        "*Note: refine will charge more credits than map-reduce.*",
        ('map-reduce', 'refine')
    )
    if chain_choice_input:
        st.success(f"✅ You selected: **{chain_choice_input}**")
    st.markdown("---")

    ### Process ###
    if st.button("Press once and wait!"):
        st.text("Processing your selections... Please wait. 😄😁😆😅😂🤣")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            progress_bar.progress(percent_complete + 1)

        # Your existing code here for processing
        docs = pdf_loader(temp_pdf_path)

        # Summarize the document
        if (model_choice_input == 'gpt-3.5-turbo-1106') & (chain_choice_input == 'map-reduce'):
            sum1 = map_reduce_result(docs, llm3)
        elif (model_choice_input == 'gpt-3.5-turbo-1106') & (chain_choice_input == 'refine'):
            sum1 = refine_result(docs, llm3)
        elif (model_choice_input == 'gpt-4-1106-preview') & (chain_choice_input == 'map-reduce'):
            sum1 = map_reduce_result(docs, llm4)
        elif (model_choice_input == 'gpt-4-1106-preview') & (chain_choice_input == 'refine'):
            sum1 = refine_result(docs, llm4)

        # Translate the summarize
        if (model_choice_input == 'gpt-3.5-turbo-1106'):
            sum2 = translate_to_thai(sum1, llm3)
        elif (model_choice_input == 'gpt-4-1106-preview'):
            sum2 = translate_to_thai(sum1, llm4)

        # After processing is complete, show the success message
        st.success(sum1)
        st.success(sum2)