import os
import openai
from langchain.chat_models import ChatOpenAI

from source_loader import pdf_loader, word_loader, powerpoint_loader, web_loader, youtube_loader, text_loader
from summarizer import map_reduce_bullet, map_reduce_paragraph, refine_bullet, refine_paragraph, translate_to_thai

import streamlit as st
import tempfile

import time


########################## Sidebar ##########################

# Set API keys
st.sidebar.title("Set API keys")

# API key input with a key
api_key_input = st.sidebar.text_input('*Note: Enter only OpenAI API key.*')
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    openai.api_key = os.environ['OPENAI_API_KEY']
    llm3 = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo-1106")
    llm4 = ChatOpenAI(temperature=0.3, model_name="gpt-4-1106-preview")
    st.sidebar.success(f"✔️ API: **{api_key_input}**")
else:
    st.sidebar.warning("Please enter your API key to proceed.")

# Model choice
st.sidebar.title("Choose model")
model_choice_input = st.sidebar.radio("*Note: GPT-4 will charge more credits.*",
                                      ('gpt-3.5-turbo-1106', 'gpt-4-1106-preview'))
if model_choice_input:
    st.sidebar.success(f"✔️ Model: **{model_choice_input}**")

# Chain choice
st.sidebar.title("Choose execution chain")
chain_choice_input = st.sidebar.radio("*Note: You can choose only map-reduce at this moment.*",
                                      ('map-reduce', 'refine','best representation vector (BRV)')
                                      )
if chain_choice_input:
    st.sidebar.success(f"✔️ Execution chain: **{chain_choice_input}**")

# Contact information
st.sidebar.title("Need help?")
st.sidebar.info("If you have any questions, please contact **Peerapat.t.**")



tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📚 PDF", "📃 Word", "📊 Powerpoint", "🌍 Website", "📺 Youtube", "✍️ Just text"])


########################## Tab 1 ##########################

with tab1:

    # Upload file
    st.header("📚 Upload PDF file")

    uploaded_file_input = st.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", 
                                           type=['pdf'], 
                                           accept_multiple_files=False, 
                                           key='uploaded_file_1')
    
    if uploaded_file_input:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file_input.read())
            temp_pdf.flush()
            temp_pdf_path = temp_pdf.name
        st.success(f"✔️ You selected: **{temp_pdf_path}**")

    st.markdown("---")

    # Select message style
    st.header("Select message style")

    st.markdown("""*🚩 **As Bullet Points:** This summary breaks down the main ideas or facts into easily digestible bullet points.\
                It's great for presentations, notes, or when you need to highlight specific key points.*""")   
    st.markdown("""*🚩 **As a Paragraph:** This is a traditional summary that condenses the main ideas into a cohesive,\
                brief narrative. It's ideal for general overviews and academic purposes.*""")
    
    messge_type_input = st.radio("*Note: Choose only one style; if unsure, you can read the instructions first.*",
                                   ('Bullet Points', 'Paragraph'), key='message_type_1')
    
    
    if messge_type_input:
        st.success(f"✔️ Message type: **{messge_type_input}**")
    st.markdown("---")


    # Process
    if st.button("🤹 Press once and wait!", key='done_1'):
        st.text("Processing your selections... Please wait.")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            progress_bar.progress(percent_complete + 1)

        docs = pdf_loader(temp_pdf_path)


        if messge_type_input == 'Bullet Points':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_bullet(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_bullet(docs, llm4)                

        if messge_type_input == 'Paragraph':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_paragraph(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_paragraph(docs, llm4)

        # Translate the summarize
        if (model_choice_input == 'gpt-3.5-turbo-1106'):
            sum2 = translate_to_thai(sum1, llm3)
        elif (model_choice_input == 'gpt-4-1106-preview'):
            sum2 = translate_to_thai(sum1, llm4)

        st.success(sum1)
        st.success(sum2)

########################## Tab 2 ##########################
        
with tab2:

    # Upload file
    st.header("📃 Upload WORD file")

    uploaded_file_input = st.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", 
                                           type=['docx'], 
                                           accept_multiple_files=False, 
                                           key='uploaded_file_2')
    
    if uploaded_file_input:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
            temp_docx.write(uploaded_file_input.read())
            temp_docx.flush()
            temp_docx_path = temp_docx.name
        st.success(f"✔️ You selected: **{temp_docx_path}**")

    st.markdown("---")

    # Select message style
    st.header("Select message style")

    st.markdown("""*🚩 **As Bullet Points:** This summary breaks down the main ideas or facts into easily digestible bullet points.\
                It's great for presentations, notes, or when you need to highlight specific key points.*""")   
    st.markdown("""*🚩 **As a Paragraph:** This is a traditional summary that condenses the main ideas into a cohesive,\
                brief narrative. It's ideal for general overviews and academic purposes.*""")
    
    messge_type_input = st.radio("*Note: Choose only one style; if unsure, you can read the instructions first.*",
                                   ('Bullet Points', 'Paragraph'), key='message_type_2')
    
    
    if messge_type_input:
        st.success(f"✔️ Message type: **{messge_type_input}**")
    st.markdown("---")


    # Process
    if st.button("🤹 Press once and wait!", key='done_2'):
        st.text("Processing your selections... Please wait.")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            progress_bar.progress(percent_complete + 1)

        docs = word_loader(temp_pdf_path)

        if messge_type_input == 'Bullet Points':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_bullet(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_bullet(docs, llm4)                

        if messge_type_input == 'Paragraph':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_paragraph(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_paragraph(docs, llm4)

        # Translate the summarize
        if (model_choice_input == 'gpt-3.5-turbo-1106'):
            sum2 = translate_to_thai(sum1, llm3)
        elif (model_choice_input == 'gpt-4-1106-preview'):
            sum2 = translate_to_thai(sum1, llm4)

        st.success(sum1)
        st.success(sum2)


########################## Tab 3 ##########################
        
with tab3:

    # Upload file
    st.header("📊 Upload POWERPOINT file")

    uploaded_file_input = st.file_uploader("*Note: Choose only 1 file to upload, larger files will charges more credit.*", 
                                           type=['pptx'], 
                                           accept_multiple_files=False, 
                                           key='uploaded_file_3')
    
    if uploaded_file_input:
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_pptx:
            temp_pptx.write(uploaded_file_input.read())
            temp_pptx.flush()
            temp_pptx_path = temp_pptx.name
        st.success(f"✔️ You selected: **{temp_pptx_path}**")

    st.markdown("---")

    # Select message style
    st.header("Select message style")

    st.markdown("""*🚩 **As Bullet Points:** This summary breaks down the main ideas or facts into easily digestible bullet points.\
                It's great for presentations, notes, or when you need to highlight specific key points.*""")   
    st.markdown("""*🚩 **As a Paragraph:** This is a traditional summary that condenses the main ideas into a cohesive,\
                brief narrative. It's ideal for general overviews and academic purposes.*""")
    
    messge_type_input = st.radio("*Note: Choose only one style; if unsure, you can read the instructions first.*",
                                   ('Bullet Points', 'Paragraph'), key='message_type_3')
    
    
    if messge_type_input:
        st.success(f"✔️ Message type: **{messge_type_input}**")
    st.markdown("---")


    # Process
    if st.button("🤹 Press once and wait!", key='done_3'):
        st.text("Processing your selections... Please wait.")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            progress_bar.progress(percent_complete + 1)

        docs = word_loader(temp_pdf_path)

        if messge_type_input == 'Bullet Points':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_bullet(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_bullet(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_bullet(docs, llm4)                

        if messge_type_input == 'Paragraph':
            if chain_choice_input == 'map-reduce':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = map_reduce_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = map_reduce_paragraph(docs, llm4)
            elif chain_choice_input == 'refine':
                if model_choice_input == 'gpt-3.5-turbo-1106':
                    sum1 = refine_paragraph(docs, llm3)
                elif model_choice_input == 'gpt-4-1106-preview':
                    sum1 = refine_paragraph(docs, llm4)

        # Translate the summarize
        if (model_choice_input == 'gpt-3.5-turbo-1106'):
            sum2 = translate_to_thai(sum1, llm3)
        elif (model_choice_input == 'gpt-4-1106-preview'):
            sum2 = translate_to_thai(sum1, llm4)

        st.success(sum1)
        st.success(sum2)