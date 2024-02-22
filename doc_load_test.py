from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import UnstructuredFileLoader

import numpy as np
import pytesseract
from pdf2image import convert_from_path

def pdf_ocr(fname, **kwargs):
	images = convert_from_path(fname, **kwargs)
	text = ''
	for img in images:
		img = np.array(img)
		text += pytesseract.image_to_string(img, lang='chi_sim')
	return text

# doc_path="/Users/ywu/Downloads/2017 雪佛兰全新迈锐宝保修及保养手册.pdf"
doc_path="/Users/ywu/Downloads/5.2.pdf"

# loader = PDFMinerLoader(doc_path)
# data = loader.load()[0]
# print(data)

# loader1 = UnstructuredFileLoader(doc_path)
# data = loader1.load()[0]
# print(data)

# loader1 = UnstructuredFileLoader(doc_path, mode="single")
# data = loader1.load()[0]
# print(data)

# text = pdf_ocr(doc_path, first_page=13, last_page=14)
text = pdf_ocr(doc_path)
# print(text)
with open("/Users/ywu/Downloads/5.2.txt", "w") as f:
	f.write(text)