import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Sidebar contents
with st.sidebar:
    st.title('This is a Talk to PDF App')
    st.markdown('''
    The Talk to PDF web application, powered by OpenAI's language model and the LangChain technology, is a compact yet powerful tool for converting spoken words into written text and generating PDF documents. With its user-friendly interface accessible through a web browser, this application offers a seamless experience across devices and platforms.

Utilizing OpenAI's advanced language model and the LangChain technology, the Talk to PDF web app accurately transcribes audio recordings or real-time conversations into text. The integration of these cutting-edge technologies ensures high transcription accuracy, capturing even subtle speech nuances.

The web application simplifies the process by allowing users to upload audio files or directly convert spoken words into text using the built-in speech recognition functionality. The intuitive interface guides users through the conversion process and provides real-time feedback on the transcription progress.

Customization options are available to format and organize the content within the PDF document. Users can choose font styles, sizes, colors, and insert headers, footers, page numbers, and other elements to enhance document structure.

Visual elements can be added to the PDF documents, allowing users to include images, charts, or tables alongside the text-based content. The application supports various image formats, ensuring compatibility with different visual elements.

Powered by the Python programming language, the Talk to PDF web app seamlessly integrates with OpenAI's language model and the LangChain technology. This modular architecture facilitates the integration of additional features and functionalities, tailored to meet specific user requirements.
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Manish')
 
load_dotenv()
 
def main():
    st.header("Talk to PDF")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()