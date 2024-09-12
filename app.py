import streamlit as st
from streamlit import session_state as session
from streamlit_pdf_viewer import pdf_viewer
from io import StringIO

if 'pdf_ref' not in session:
    session.pdf_ref = None

st.logo('src/webapp/static/pepsi.png')
st.sidebar.markdown("Hi!")
container = st.container(border=True)
container.write('Demonstration powered by Kodio AI Software')
uploaded_file = st.file_uploader("Select a file to work with")
session.pdf_ref = uploaded_file

if uploaded_file is not None:
    # To read file as bytes:

    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=700)
    # st.write(bytes_data)

    # with open('doc.pdf', 'wb') as f:
    #     f.write(binary_data)

st.chat_input('Ask a question')
