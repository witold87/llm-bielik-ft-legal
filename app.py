import time
import streamlit as st

from streamlit import session_state as session
from streamlit_pdf_viewer import pdf_viewer

from src.document_processor import DocumentProcessor
from src.doc_store.doc_stores import FaissDocStore
from src.inference.lexio_model import LexioModel

if 'pdf_ref' not in session:
    session.pdf_ref = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if session.pdf_ref is None:
    st.session_state.messages = []

st.logo('src/webapp/static/pepsi.png')
st.sidebar.markdown("Lexio / Lexai")

st.title('Lexio by Kodio AI')
st.divider()
container = st.container()
container.write('Demonstration powered by Kodio AI Software')
st.divider()
uploaded_file = st.file_uploader("Select a file to work with")
session.pdf_ref = uploaded_file


def stream_data(answer: str):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.02)

if session.pdf_ref is not None:
    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=700)

    extracted_text = DocumentProcessor(chunk_size=512, file=binary_data).extract_and_split()
    doc_store = FaissDocStore(text=extracted_text)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Rozpocznij rozmowę..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        storage_results = doc_store.index_search(query=prompt)

        model_output = LexioModel().run_inference(query=prompt, context=storage_results)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(model_output)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": model_output})


    #
    # if st.button('Show answer'):
    #     st.write_stream(stream_data(single_res))

