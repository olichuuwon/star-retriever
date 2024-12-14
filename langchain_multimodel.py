import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from base64 import b64decode
import uuid

# Load environment variables from .env
load_dotenv()

# Set API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Missing the OPENAI_API_KEY in the .env file.")

# Paths
output_path = "./content/"
file_path = os.path.join(output_path, "attention.pdf")

# Extract the PDF data
try:
    print(f"Processing PDF file: {file_path}")
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    print(f"Extracted {len(chunks)} chunks from the PDF.")
except Exception as e:
    print(f"Error while processing PDF: {e}")
    exit(1)

# Separate elements into tables, text, and images
tables, texts = [], []
for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

print(f"Found {len(tables)} tables and {len(texts)} text elements.")


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


images = get_images_base64(chunks)
print(f"Extracted {len(images)} images as base64.")

# Summarize text and tables
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text, including any page references.
Table or text chunk: {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

print("Generating summaries for text elements...")
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
print("Generating summaries for table elements...")
table_summaries = summarize_chain.batch(
    [table.metadata.text_as_html for table in tables], {"max_concurrency": 3}
)

# Summarize images
image_prompt_template = """Describe the image in detail. For context, 
the image is part of a research paper explaining the transformers architecture. Be specific about graphs, such as bar plots."""
image_chain = (
    {"image": lambda x: x}
    | ChatPromptTemplate.from_template(image_prompt_template)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)
print("Generating summaries for images...")
image_summaries = image_chain.batch(images)

# Load data into vectorstore
embedding_function = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="multi_modal_rag", embedding_function=embedding_function
)
store = InMemoryStore()
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=store, id_key="doc_id"
)


# Link summaries to original data
def add_to_retriever(summaries, originals, retriever, element_type):
    doc_ids = [str(uuid.uuid4()) for _ in originals]
    docs = [
        Document(
            page_content=summary, metadata={"doc_id": doc_ids[i], "type": element_type}
        )
        for i, summary in enumerate(summaries)
    ]
    # Ensure docs are only added if embeddings are non-empty
    if docs:
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, originals)))
        print(f"Added {len(docs)} {element_type} documents to retriever.")


add_to_retriever(text_summaries, texts, retriever, "text")
add_to_retriever(table_summaries, tables, retriever, "table")
add_to_retriever(image_summaries, images, retriever, "image")


# Define RAG pipeline
def parse_docs(docs):
    """Split base64-encoded images and texts."""
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join(doc.text for doc in docs_by_type["texts"])

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]
    for image in docs_by_type["images"]:
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

print("Generating response to the question...")
response = chain.invoke("What is the attention mechanism?")
print("Response:")
print(response)
