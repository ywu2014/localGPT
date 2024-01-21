from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("/Users/ywu/Downloads/nl2sql - wms - 多表测试.pdf")
data = loader.load()[0]
print(data)