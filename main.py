import os
import re
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Quiz Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
embedding_model = None

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'Halaman\s+\d+(\s+dari\s+\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s.,?!:;\-()/"\'%-]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return " ".join(lines)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/ingest")
async def process_pdf_files(rps_file: UploadFile = File(...), modul_file: UploadFile = File(...)):
    global vector_store, embedding_model

    all_documents = []
    files_to_process = [("RPS", rps_file), ("Modul", modul_file)]

    for doc_type, file_obj in files_to_process:
        with pdfplumber.open(file_obj.file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                cleaned_text = clean_text(raw_text)
                if cleaned_text:
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "doc_type": doc_type,
                            "source": file_obj.filename,
                            "page": page_num
                        }
                    )
                    all_documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    vector_store = FAISS.from_documents(chunks, embedding_model)

    return {
        "status": "success",
        "message": f"✅ Berhasil memproses {len(all_documents)} halaman PDF ke dalam {len(chunks)} segmen FAISS Vector DB."
    }

@app.post("/api/generate-quiz")
async def generate_quiz(
    topic_query: str = Form(...),
    num_questions: int = Form(...),
    question_format: str = Form(...),
    bloom_level: str = Form(...)
):
    global vector_store
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY belum dikonfigurasi di server")
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Unggah dan indeks dokumen RPS & Modul terlebih dahulu!")

    client = genai.Client(api_key=gemini_api_key)
    retrieved_docs = vector_store.similarity_search(topic_query, k=5)

    context_text = ""
    for doc in retrieved_docs:
        context_text += f"\n--- [Sumber: {doc.metadata['doc_type']} | Halaman: {doc.metadata['page']}] ---\n{doc.page_content}\n"

    bloom_desc = {
        "Mudah (C1-C2)": "C1 (Remembering) dan C2 (Understanding)",
        "Sedang (C3-C4)": "C3 (Applying) dan C4 (Analyzing)",
        "Sukar (C5-C6)": "C5 (Evaluating) dan C6 (Creating)"
    }

    prompt_template = f"""
    Anda adalah seorang dosen dan pakar evaluasi akademik di perguruan tinggi.
    Tugas Anda adalah membuat instrumen kuis/soal evaluasi mahasiswa secara akurat berdasarkan KONTEKS TEKS RUJUKAN yang diberikan di bawah ini.

    DILARANG KERAS MENGGUNAKAN INFORMASI DI LUAR KONTEKS TEKS RUJUKAN (BEBAS HALUSINASI).

    === KONTEKS TEKS RUJUKAN (RPS & MODUL) ===
    {context_text}

    === PARAMETER INSTRUKSI PENYUSUNAN SOAL ===
    - Topik Materi: {topic_query}
    - Jumlah Soal: {num_questions} butir
    - Format Soal: {question_format}
    - Tingkat Kesulitan (Taksonomi Bloom): {bloom_level} -> {bloom_desc.get(bloom_level, '')}

    === FORMAT OUTPUT ===
    1. Jika Format "Pilihan Ganda":
       - Sajikan soal yang jelas.
       - Berikan 4 pilihan jawaban (A, B, C, D).
       - Sertakan Kunci Jawaban dan Pembahasan Singkat berbasis konteks rujukan.
    2. Jika Format "Esai":
       - Sajikan pertanyaan esai yang mendalam.
       - Sertakan Rubrik Penilaian / Kunci Jawaban Acuan berbasis konteks rujukan.

    Sajikan output secara rapi, profesional, dan terstruktur.
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt_template
    )

    return {"status": "success", "quiz_result": response.text}
