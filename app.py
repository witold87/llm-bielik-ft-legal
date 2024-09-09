import streamlit as st
from io import StringIO


st.logo('src/webapp/static/pepsi.png')
st.sidebar.markdown("Hi!")
container = st.container(border=True)
container.write('Demo for Pravna powered by Kodio AI Software')
uploaded_file = st.file_uploader("Select a file to work with")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

st.chat_input('Ask a question')
