from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_LENGTH = 40


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [
        chunk for chunk in splitter.split_documents(documents)
        if len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH
    ]
