from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import urllib.parse
import os
# pip install rapidocr-onnxruntime
# pip install unstructured


def pdf_loader(pdf_path):
    # loader = PyPDFLoader(pdf_path, extract_images=True)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text = ''
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs


def word_loader(word_path):
    loader = Docx2txtLoader(word_path)
    pages = loader.load()

    text = ''
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs


def powerpoint_loader(powerpoint_path):
    loader = UnstructuredPowerPointLoader(powerpoint_path)
    pages = loader.load()

    text = ''
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs


def web_loader(web_url):
    loader = WebBaseLoader(web_url)
    pages = loader.load()

    text = ''
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs


def youtube_loader(yt_url):
    # ref from github.com/e-johnstonn
    parsed_url = urllib.parse.urlparse(yt_url)
    final_url = ''
    if parsed_url.hostname == 'youtu.be':
        final_url = parsed_url.path[1:]

    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            p = urllib.parse.parse_qs(parsed_url.query)
            final_url = p.get('v', [None])[0]

        elif parsed_url.path.startswith('/embed/'):
            final_url = parsed_url.path.split('/embed/')[1]

        elif parsed_url.path.startswith('/v/'):
            final_url = parsed_url.path.split('/v/')[1]

    loader = YoutubeLoader(final_url)
    pages = loader.load()

    text = ''
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs


def text_loader(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text])

    return docs