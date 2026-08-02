try:
    from langchain_community.document_loaders import JSONLoader
except ImportError:
    from langchain.document_loaders import JSONLoader  # type: ignore[import-not-found]


def metadata_func(record: dict, metadata: dict) -> dict:
    """Attach useful fields as metadata instead of dumping them into the text."""
    metadata["id"] = record.get("id")
    metadata["source_url"] = record.get("source_url")
    metadata["title"] = record.get("title")
    return metadata


def load_university_data(json_path: str = "data/old_seu_website_cleaned.jsonl"):
    """
    Loads university JSONL data into LangChain Document objects.
    Prepends the title to the content so the title's keywords contribute
    to retrieval matching (e.g. a query about "vision" should strongly
    match a record titled "Vision").
    """
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".",
        content_key="clean_text",
        metadata_func=metadata_func,
        json_lines=True,
        text_content=True,
    )
    docs = loader.load()

    # Prepend title to content for better semantic matching
    for doc in docs:
        title = doc.metadata.get("title", "")
        if title:
            doc.page_content = f"{title}\n\n{doc.page_content}"

    return docs


if __name__ == "__main__":
    docs = load_university_data()
    print(f"Loaded {len(docs)} documents")
    if docs:
        print("Sample content:\n", docs[0].page_content[:200])
        print("Sample metadata:\n", docs[0].metadata)