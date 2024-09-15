from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from PyPDF2 import PdfReader


class DocumentProcessor:
    def __init__(self, file: bytes, chunk_size: int = 512):
        self.file = file
        self.chunk_size= chunk_size
        self.text = ''

    def extract_and_split(self):
        self._extract_text()
        return self.split_into_chunks()

    def _extract_text(self) -> str:
        pdf_stream = BytesIO(self.file)
        pdf_reader = PdfReader(pdf_stream)
        extracted_text = str()
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
        self.text = extracted_text
        return extracted_text

    def split_into_chunks(self) -> list[str]:
        overlap = self._calc_overlap()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=overlap)
        return text_splitter.split_text(self.text)


    def _calc_overlap(self, overlap_perc: float = 0.25) -> int:
        return int(self.chunk_size*overlap_perc)

