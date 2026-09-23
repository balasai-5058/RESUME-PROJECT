import streamlit as st
import pdfplumber
import chromadb

st.set_page_config(page_title="DocuChat", layout="wide")
st.title("📄 DocuChat — Ask Your Documents")

from groq import Groq

groq_api_key = st.secrets["GROQ_API_KEY"]
client_groq = Groq(api_key=groq_api_key)

@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client()
    return client.get_or_create_collection(name="documents")

collection = get_chroma_collection()

def extract_text_from_pdf(pdf_file):
    pages_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_data.append({"page_number": i + 1, "text": text})
    return pages_data

def chunk_text(pages_data, chunk_size=100, overlap=20):
    chunks = []
    for page in pages_data:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append({"text": " ".join(chunk_words), "page_number": page["page_number"]})
            start += chunk_size - overlap
    return chunks

def add_chunks_to_db(chunks, doc_name):
    texts = [c["text"] for c in chunks]
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page_number"], "source": doc_name} for c in chunks]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)

def retrieve_relevant_chunks(question, top_k=4):
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "page": meta["page"], "source": meta["source"]})
    return retrieved

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in retrieved_chunks])
    prompt = f"""You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer ONLY using the context above.
- If the answer isn't in the context, say "I couldn't find that in the document."
- Cite the page number(s) you used.

Answer:"""
    response = client_groq.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.button("Process Document"):
        with st.spinner("Reading and indexing document..."):
            pages = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(pages)
            add_chunks_to_db(chunks, doc_name=uploaded_file.name)
            st.session_state.processed_docs.append(uploaded_file.name)
        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

if st.session_state.processed_docs:
    st.caption(f"Documents indexed: {', '.join(st.session_state.processed_docs)}")

st.divider()

question = st.text_input("Ask a question about your document:")
if question:
    if not st.session_state.processed_docs:
        st.warning("Please upload and process a document first.")
    else:
        with st.spinner("Searching and generating answer..."):
            chunks_found = retrieve_relevant_chunks(question)
            answer = generate_answer(question, chunks_found)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for c in chunks_found:
                st.write(f"**Page {c['page']}:** {c['text'][:200]}...")
