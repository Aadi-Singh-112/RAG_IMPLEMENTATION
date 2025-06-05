import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import io  # <--- For in-memory text download

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY']='AIzaSyADB0m6DljZH-GqTyevymT4ax9NhBaQ9RI'

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

def ask_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error: {e}. Please process PDFs first."

    docs = db.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = """
    Answer the question using the context below. Be accurate and do not make up information.
    If the answer is not in the context, say: "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    final_prompt = prompt.format(context=context, question=question)

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    response = model.invoke(final_prompt)
    return response.content.strip()

# --- Streamlit App ---
def main():
    st.set_page_config("PDF Q&A", page_icon="📘", layout="centered")

    st.markdown("""
        <style>
            body {
                background-color: #f9f9f9;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            h1 {
                color: #2C3E50;
                font-weight: 700;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            .question-box, .answer-box {
                background-color: #fff;
                padding: 1rem 1.2rem;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                margin-bottom: 1.5rem;
            }
            .question-box {
                border-left: 5px solid #3498DB;
            }
            .answer-box {
                border-left: 5px solid #2ECC71;
            }
            .stTextInput > div > div > input {
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border: 1px solid #ccc;
            }
            .stButton > button {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 0.6rem 1.2rem;
                border-radius: 8px;
                font-weight: 600;
                transition: 0.2s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #2980B9;
                transform: translateY(-1px);
            }
            .sidebar .stButton > button {
                width: 100%;
            }
            hr {
                border: none;
                border-top: 1px solid #ddd;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📘 Ask Questions About Your PDFs")
    st.markdown("Upload PDFs on the sidebar, ask questions below, and download all answers.")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files:", accept_multiple_files=True, type="pdf")
        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        chunks = get_text_chunks(raw_text)
                        get_vector_store(chunks)
                        st.success("✅ PDFs processed successfully.")
                        st.session_state.qa_history.clear()
                    else:
                        st.warning("No text could be extracted from uploaded files.")
            else:
                st.warning("Please upload at least one PDF.")

    question = st.text_input("Ask a question:", placeholder="e.g., What is the document about?")
    if st.button("🔍 Get Answer"):
        if question:
            with st.spinner("Searching..."):
                answer = ask_question(question)
            st.session_state.qa_history.append((question, answer))
        else:
            st.warning("Please enter a question.")

    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("### 🧾 Question & Answer History")
        for idx, (q, a) in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"<div class='question-box'><strong>Q{idx}:</strong> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'><strong>Answer:</strong> {a}</div>", unsafe_allow_html=True)

        # Prepare downloadable .txt content
        output = io.StringIO()
        for idx, (q, a) in enumerate(st.session_state.qa_history, 1):
            output.write(f"Q{idx}: {q}\nA{idx}: {a}\n\n")
        txt_bytes = output.getvalue().encode("utf-8")

        st.download_button(
            label="📥 Download Q&A History (.txt)",
            data=txt_bytes,
            file_name="qa_history.txt",
            mime="text/plain",
        )

    st.markdown("""
        <hr/>
        <div style='text-align:center; font-size: 13px; color: #777;'>
            Developed by <a href='https://github.com/Aadi-Singh-112' target='_blank'>Aditya Singh</a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
