import os
import re
import pdfplumber
import gradio as gr
from google import genai
from google.genai import types

# Impor pustaka LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================================================================
# GLOBAL VARIABLES FOR VECTOR STORE & EMBEDDING MODEL
# ==============================================================================
vector_store = None
embedding_model = None

# ==============================================================================
# TAHAP 1: DATA INGESTION & DATA CLEANING
# ==============================================================================
def clean_text(text):
    """Pembersihan teks dari header/footer, simbol aneh, dan spasi berlebih."""
    if not text:
        return ""
    text = re.sub(r'Halaman\s+\d+(\s+dari\s+\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s.,?!:;\-()/"\'%-]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return " ".join(lines)

def process_pdf_files(rps_file, modul_file):
    """Ekstraksi teks dari file RPS dan Modul ke format Document LangChain."""
    global vector_store, embedding_model

    if not rps_file or not modul_file:
        return "⚠️ Harap unggah KEDUA dokumen (RPS dan Modul Perkuliahan) terlebih dahulu!"

    all_documents = []
    files_to_process = [("RPS", rps_file), ("Modul", modul_file)]

    for doc_type, file_obj in files_to_process:
        # Menangani kompatibilitas path file di Gradio
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        file_name = os.path.basename(file_path)

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                cleaned_text = clean_text(raw_text)

                if cleaned_text:
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "doc_type": doc_type,
                            "source": file_name,
                            "page": page_num
                        }
                    )
                    all_documents.append(doc)

    # ==============================================================================
    # TAHAP 2: CHUNKING & EMBEDDING
    # ==============================================================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Memuat Embedding Model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    # ==============================================================================
    # TAHAP 3: VECTOR DATABASE INDEXING (FAISS)
    # ==============================================================================
    vector_store = FAISS.from_documents(chunks, embedding_model)

    return f"✅ Berhasil memproses dokumen!\n- Total Dokumen Ingesti: {len(all_documents)} halaman\n- Total Chunks Diindeks: {len(chunks)} segmen ke FAISS VectorDB."

# ==============================================================================
# TAHAP 4 & 5: RETRIEVAL & CONTROLLED GENERATION
# ==============================================================================
def generate_quiz(topic_query, num_questions, question_format, bloom_level):
    global vector_store

    # Membaca GEMINI_API_KEY dari Environment Variable sistem/cloud
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not gemini_api_key:
        return "⚠️ API Key tidak ditemukan! Harap pastikan variabel 'GEMINI_API_KEY' telah dikonfigurasi di Environment Variables server Anda."
    if vector_store is None:
        return "⚠️ Harap unggah dan proses dokumen RPS & Modul terlebih dahulu pada Tab 1!"
    if not topic_query.strip():
        return "⚠️ Harap masukkan topik perkuliahan yang ingin dibuatkan soal!"

    # 1. Konfigurasi Client Gemini SDK
    try:
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        return f"Error konfigurasi Gemini API Client: {str(e)}"

    # 2. Deteksi Model Gemini yang Aktif
    active_models = []
    try:
        for m in client.models.list():
            model_id = m.name.replace("models/", "") if hasattr(m, 'name') else str(m)
            if "gemini" in model_id.lower():
                supported_methods = getattr(m, 'supported_generation_methods', [])
                if not supported_methods or "generateContent" in supported_methods:
                    active_models.append(model_id)
    except Exception as e:
        active_models = ['gemini-2.0-flash', 'gemini-1.5-flash-002', 'gemini-1.5-flash']

    if not active_models:
        active_models = ['gemini-2.0-flash', 'gemini-1.5-flash-002', 'gemini-1.5-flash']

    # TAHAP 4: RETRIEVAL COMPONENT (Search Cosine Similarity)
    retrieved_docs = vector_store.similarity_search(topic_query, k=5)

    context_text = ""
    for doc in retrieved_docs:
        context_text += f"\n--- [Sumber: {doc.metadata['doc_type']} | Halaman: {doc.metadata['page']}] ---\n{doc.page_content}\n"

    # Mapping Deskripsi Taksonomi Bloom
    bloom_desc = {
        "Mudah (C1-C2)": "C1 (Remembering/Mengingat) dan C2 (Understanding/Memahami). Soal berfokus pada definisi, konsep dasar, dan pemahaman faktual.",
        "Sedang (C3-C4)": "C3 (Applying/Menerapkan) dan C4 (Analyzing/Menganalisis). Soal berfokus pada studi kasus, penerapan rumus/konsep, serta analisis hubungan antar konsep.",
        "Sukar (C5-C6)": "C5 (Evaluating/Mengevaluasi) dan C6 (Creating/Menciptakan). Soal berfokus pada penilaian kritis, pemecahan masalah kompleks, serta merancang solusi/ide baru."
    }

    # TAHAP 5: CONTROLLED GENERATION
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

    last_error = ""
    for model_name in active_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_template
            )
            return response.text
        except Exception as e:
            last_error = str(e)
            continue

    return f"⚠️ Gagal melakukan generasi soal.\n\nDetail Error Terakhir: {last_error}"

# ==============================================================================
# GRADIO INTERFACE BUILDER
# ==============================================================================
with gr.Blocks(title="Generator Soal Cerdas Berbasis RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎓 Generator Soal Cerdas Berbasis RAG (Retrieval-Augmented Generation)
        **Otomatisasi Evaluasi Mahasiswa Terintegrasi RPS dan Modul Perkuliahan**
        """
    )

    with gr.Tab("📁 Tab 1: Ingesti & Indeks Dokumen"):
        gr.Markdown("### Unggah Dokumen Akademik (PDF)")
        with gr.Row():
            rps_input = gr.File(label="Upload PDF RPS (Rencana Pembelajaran Semester)", file_types=[".pdf"])
            modul_input = gr.File(label="Upload PDF Modul Perkuliahan", file_types=[".pdf"])

        process_btn = gr.Button("🚀 Proses & Indeks Dokumen (FAISS)", variant="primary")
        status_output = gr.Textbox(label="Status Ingesti Data", interactive=False, lines=4)

        process_btn.click(
            fn=process_pdf_files,
            inputs=[rps_input, modul_input],
            outputs=status_output
        )

    with gr.Tab("⚙️ Tab 2: Generasi Soal Cerdas"):
        gr.Markdown("### Parameter Kontrol Pembuatan Kuis")

        with gr.Row():
            topic_input = gr.Textbox(
                label="Topik / Sub-Capaian Perkuliahan",
                placeholder="Contoh: Konsep Dasar Database Relasional / Network Security",
                scale=2
            )
            num_input = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Jumlah Soal")

        with gr.Row():
            format_input = gr.Radio(
                choices=["Pilihan Ganda", "Esai"],
                value="Pilihan Ganda",
                label="Format Soal"
            )
            bloom_input = gr.Dropdown(
                choices=["Mudah (C1-C2)", "Sedang (C3-C4)", "Sukar (C5-C6)"],
                value="Sedang (C3-C4)",
                label="Tingkat Kesulitan (Taksonomi Bloom)"
            )

        generate_btn = gr.Button("✨ Generasi Soal Bebas Halusinasi", variant="primary")
        quiz_output = gr.Markdown(label="Hasil Generasi Soal")

        generate_btn.click(
            fn=generate_quiz,
            inputs=[topic_input, num_input, format_input, bloom_input],
            outputs=quiz_output
        )

# ==============================================================================
# LAUNCH APPLICATION WITH DYNAMIC PORT MANAGEMENT
# ==============================================================================
if __name__ == "__main__":
    # Menyesuaikan port dengan standar platform cloud deployment (seperti Render/Heroku/AWS)
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
