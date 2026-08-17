import os
import shutil
import re

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DOCS_PATH = "./docs"
DB_PATH = "./chroma_db"


# =========================================================
# METİN TEMİZLEME
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Sayfa numarası benzeri ifadeler
    text = re.sub(
        r"Sayfa\s*\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Fazla boşluklar
    text = re.sub(r"[ \t]+", " ", text)

    # Fazla boş satırlar
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# =========================================================
# PDF OKUMA
# =========================================================

def read_pdf(pdf_path):

    pages = []

    try:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

            except Exception as e:

                print(
                    f"⚠️ {os.path.basename(pdf_path)} "
                    f"sayfa {page_number} okunamadı: {e}"
                )

                continue

            text = clean_text(text)

            if len(text) < 20:
                continue

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_number
                    }
                )
            )

    except Exception as e:

        print(
            f"❌ HATA: "
            f"{os.path.basename(pdf_path)} -> {e}"
        )

    return pages


# =========================================================
# TÜM PDF'LERİ OKU
# =========================================================

def read_all_pdfs():

    if not os.path.exists(DOCS_PATH):

        raise FileNotFoundError(
            f"'{DOCS_PATH}' klasörü bulunamadı."
        )

    pdf_files = sorted(
        file
        for file in os.listdir(DOCS_PATH)
        if file.lower().endswith(".pdf")
    )

    if not pdf_files:

        print(
            "❌ docs klasöründe PDF bulunamadı."
        )

        return []

    print("=" * 60)
    print(f"📚 {len(pdf_files)} PDF bulundu.")
    print("=" * 60)

    all_pages = []

    for file_name in pdf_files:

        pdf_path = os.path.join(
            DOCS_PATH,
            file_name
        )

        print(
            f"\n📖 Okunuyor: {file_name}"
        )

        pages = read_pdf(pdf_path)

        if pages:

            all_pages.extend(pages)

            character_count = sum(
                len(page.page_content)
                for page in pages
            )

            print(
                f"✅ {len(pages)} sayfa okundu "
                f"| {character_count} karakter"
            )

        else:

            print(
                "⚠️ Bu PDF'den metin alınamadı."
            )

    print("\n" + "=" * 60)
    print(
        f"📊 Toplam işlenen sayfa: "
        f"{len(all_pages)}"
    )
    print("=" * 60)

    return all_pages


# =========================================================
# CHUNK OLUŞTUR
# =========================================================

def create_chunks(documents):

    if not documents:

        print(
            "❌ Parçalanacak doküman yok."
        )

        return []

    print(
        "\n✂️ Dokümanlar parçalara ayrılıyor..."
    )

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=120,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Çok küçük parçaları kaldır
    chunks = [
        chunk
        for chunk in chunks
        if len(
            chunk.page_content.strip()
        ) >= 80
    ]

    # Chunk ID
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    print(
        f"🧩 Toplam chunk: {len(chunks)}"
    )

    return chunks


# =========================================================
# VECTOR DATABASE
# =========================================================

def build_vector_db():

    print("=" * 60)
    print("📚 İLK YARDIM PDF'LERİ İŞLENİYOR")
    print("=" * 60)

    # -----------------------------------------------------
    # Eski Chroma'yı sil
    # -----------------------------------------------------

    if os.path.exists(DB_PATH):

        print(
            "\n🗑️ Eski Chroma veritabanı siliniyor..."
        )

        shutil.rmtree(DB_PATH)

    # -----------------------------------------------------
    # PDF'leri oku
    # -----------------------------------------------------

    documents = read_all_pdfs()

    if not documents:

        print(
            "\n❌ HİÇBİR PDF OKUNAMADI."
        )

        return

    # -----------------------------------------------------
    # Chunk
    # -----------------------------------------------------

    chunks = create_chunks(documents)

    if not chunks:

        print(
            "\n❌ HİÇ CHUNK OLUŞTURULAMADI."
        )

        return

    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    print(
        "\n🔢 Türkçe embedding modeli yükleniyor..."
    )

    embeddings = HuggingFaceEmbeddings(

        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    print(
        "✅ Türkçe embedding modeli hazır."
    )

    # -----------------------------------------------------
    # Chroma
    # -----------------------------------------------------

    print(
        "\n💾 Chroma veritabanı oluşturuluyor..."
    )

    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=DB_PATH
    )

    print("\n" + "=" * 60)
    print(
        "✅ VERİTABANI BAŞARIYLA OLUŞTURULDU!"
    )

    print(
        f"📊 PDF sayfası : {len(documents)}"
    )

    print(
        f"🧩 Chunk sayısı: {len(chunks)}"
    )

    print(
        f"📁 Veritabanı  : {DB_PATH}"
    )

    print("=" * 60)


# =========================================================
# BAŞLAT
# =========================================================

if __name__ == "__main__":

    build_vector_db()
