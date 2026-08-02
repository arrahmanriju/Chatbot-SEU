from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src.loader import load_university_data
except ModuleNotFoundError:
    from loader import load_university_data

MIN_CHUNK_LENGTH = 40  # discard tiny, low-information chunks (e.g. nav fragments)


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Splits documents into smaller overlapping chunks for better retrieval.
    Filters out chunks that are too short to carry useful information.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    return chunks


if __name__ == "__main__":
    docs = load_university_data()
    chunks = split_documents(docs)
    print(f"Original documents: {len(docs)}")
    print(f"Total chunks after splitting and filtering: {len(chunks)}")
    print("\nSample chunk:\n", chunks[0].page_content)
    print("\nSample chunk metadata:\n", chunks[0].metadata)