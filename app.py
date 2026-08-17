import os
import re
import subprocess

import streamlit as st

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# AYARLAR
# =========================================================

DB_PATH = "./chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# STREAMLIT SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Acil Yardım Rehberi",
    page_icon="🚑",
    layout="centered"
)


# =========================================================
# FOUNDRY PORTUNU OTOMATİK BUL
# =========================================================

def get_foundry_url():

    try:

        result = subprocess.run(
            ["foundry", "server", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout

        match = re.search(
            r"http://127\.0\.0\.1:(\d+)",
            output
        )

        if match:

            port = match.group(1)

            return (
                f"http://127.0.0.1:{port}/v1"
            )

    except Exception as e:

        print("Foundry port bulunamadı:", e)

    return None


# =========================================================
# FOUNDRY CLIENT
# =========================================================

@st.cache_resource
def get_foundry_client():

    foundry_url = get_foundry_url()

    if foundry_url is None:
        raise RuntimeError(
            "Foundry Local sunucusu bulunamadı."
        )

    client = OpenAI(
        base_url=foundry_url,
        api_key="foundry"
    )

    models = client.models.list()

    model_name = None

    for model in models.data:

        model_id = model.id.lower()

        if "phi-3.5-mini" in model_id:

            model_name = model.id
            break

    if model_name is None:

        raise RuntimeError(
            "phi-3.5-mini modeli Foundry Local'da bulunamadı."
        )

    return client, model_name, foundry_url


# =========================================================
# EMBEDDING
# =========================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =========================================================
# CHROMA
# =========================================================

@st.cache_resource
def get_vector_db():

    if not os.path.exists(DB_PATH):

        raise RuntimeError(
            "chroma_db klasörü bulunamadı. "
            "Önce ingest.py çalıştırılmalı."
        )

    embeddings = get_embeddings()

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vector_db


# =========================================================
# KONU TESPİTİ
# =========================================================

TOPIC_KEYWORDS = {

    "kalp_krizi": [
        "kalp krizi",
        "kalp krizi geçiriyor",
        "kalp krizi geçiren",
        "göğüs ağrısı",
        "göğüs sıkışması"
    ],

    "havale": [
        "havale",
        "nöbet geçiriyor",
        "nöbet geçiren",
        "kasılma",
        "ateşli havale"
    ],

    "bayilma": [
        "bayılma",
        "bayıldı",
        "baygın",
        "bilinci kapalı"
    ],

    "kanama": [
        "kanama",
        "kanıyor",
        "kanayan"
    ],

    "kene": [
        "kene",
        "kene yapıştı",
        "kene ısırdı"
    ],

    "yanma": [
        "yanık",
        "yanma",
        "yandı"
    ],

    "kedi_kopek_isirmasi": [
        "köpek ısırdı",
        "köpek ısırması",
        "kedi ısırdı",
        "kedi ısırması",
        "hayvan ısırması"
    ],

    "ari_akrep_yilan": [
        "arı soktu",
        "arı sokması",
        "akrep soktu",
        "akrep sokması",
        "yılan soktu",
        "yılan sokması"
    ],

    "kırık": [
        "kırık",
        "kırıldı",
        "kemik kırılması"
    ],

    "diyabet": [
        "diyabet",
        "şeker hastası",
        "kan şekeri"
    ],

    "sara_epilepsi": [
        "sara",
        "epilepsi",
        "epilepsi nöbeti"
    ],

    "yüksek_ateş_havalesi": [
        "yüksek ateş",
        "ateşli havale"
    ],

    "sarilik": [
        "sarılık"
    ]
}


# =========================================================
# KONU -> PDF
# =========================================================

TOPIC_FILES = {

    "kalp_krizi": [
        "kalp_krizi.pdf"
    ],

    "havale": [
        "havale_genel_ilk_yardim.pdf",
        "yuksek_ates_havalesi_ilk_yardim.pdf"
    ],

    "bayilma": [
        "bayilma_durumu.pdf"
    ],

    "kanama": [
        "kanama_durumu.pdf"
    ],

    "kene": [
        "kene.pdf"
    ],

    "yanma": [
        "yanma_durumu.pdf"
    ],

    "kedi_kopek_isirmasi": [
        "kedi_kopek_isirmasi_ilk_yardim.pdf"
    ],

    "ari_akrep_yilan": [
        "ari_akrep_yilan_sokmasi_ilk_yardim.pdf"
    ],

    "kırık": [
        "kirik_durumu.pdf"
    ],

    "diyabet": [
        "diyabet_ilk_yardim.pdf"
    ],

    "sara_epilepsi": [
        "sara_epilepsi_ilk_yardim.pdf"
    ],

    "yüksek_ateş_havalesi": [
        "yuksek_ates_havalesi_ilk_yardim.pdf"
    ],

    "sarilik": [
        "sarilik_ilk_yardim.pdf"
    ]
}


# =========================================================
# KONU TESPİT
# =========================================================

def detect_topic(query):

    query_lower = query.lower()

    # Daha özel konular önce kontrol edilir.
    ordered_topics = [
        "yüksek_ateş_havalesi",
        "kalp_krizi",
        "kedi_kopek_isirmasi",
        "ari_akrep_yilan",
        "sara_epilepsi",
        "havale",
        "bayilma",
        "kanama",
        "kene",
        "yanma",
        "kırık",
        "diyabet",
        "sarilik"
    ]

    for topic in ordered_topics:

        for keyword in TOPIC_KEYWORDS.get(topic, []):

            if keyword in query_lower:

                return topic

    return None


# =========================================================
# DOKÜMAN FİLTRELEME
# =========================================================

def search_documents(vector_db, query, topic):

    target_files = TOPIC_FILES.get(
        topic,
        []
    )

    if not target_files:
        return []


    # Chroma'dan daha geniş sonuç alıyoruz.
    results = vector_db.similarity_search(
        query,
        k=12
    )


    filtered = []

    for doc in results:

        source = doc.metadata.get(
            "source",
            ""
        )

        filename = os.path.basename(source)

        if filename in target_files:

            filtered.append(doc)


    return filtered


# =========================================================
# CONTEXT
# =========================================================

def create_context(docs):

    parts = []

    for index, doc in enumerate(
        docs,
        start=1
    ):

        text = doc.page_content.strip()

        if len(text) < 30:
            continue

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Bilinmeyen kaynak"
            )
        )

        page = doc.metadata.get(
            "page",
            ""
        )

        parts.append(
            f"[Kaynak {index}: {source} | Sayfa {page}]\n"
            f"{text}"
        )

    return "\n\n".join(parts)


# =========================================================
# YAPAY ZEKA CEVABI
# =========================================================

def generate_answer(
    client,
    model_name,
    query,
    context
):

    system_prompt = """
Sen "Acil Yardım Rehberi" adlı yerel RAG
sisteminin ilk yardım asistanısın.

ÇOK ÖNEMLİ:

- Yalnızca verilen PDF içeriğini kullan.
- PDF'de olmayan bilgiyi ekleme.
- Tahmin yapma.
- Konuyu değiştirme.
- Kullanıcının sormadığı başka bir hastalık
  veya durum hakkında konuşma.
- Cevabı Türkçe ver.
- Kısa ama yeterince açıklayıcı ol.
- PDF'deki ilk yardım adımlarını mümkün olduğunca
  sırayla ve anlaşılır şekilde aktar.
- Gereksiz giriş cümleleri yazma.
- "PDF metinlerinden alınan..." gibi ifadeler kullanma.
- Kullanıcıya doğrudan yardımcı ol.
- Tıbbi teşhis koyma.
- Kaynakta olmayan ilaç, doz veya tedavi önerme.
- Kaynakta 112 belirtiliyorsa 112 bilgisini aktar.

SADECE VERİLEN DOKÜMANLARDAKİ BİLGİLERLE
CEVAP VER.
"""

    user_prompt = f"""
KULLANICININ SORUSU:

{query}


İLGİLİ PDF İÇERİĞİ:

{context}


Bu soruyu yalnızca yukarıdaki PDF içeriğine
dayanarak cevapla.

Cevabı maddeler halinde ve doğrudan ver.
"""

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.0,

        max_tokens=500
    )

    return response.choices[0].message.content


# =========================================================
# STREAMLIT ARAYÜZ
# =========================================================

st.title("🚑 Acil Yardım Rehberi")

st.caption(
    "Yerel PDF dokümanları ve Foundry Local "
    "ile çalışan RAG sistemi"
)

st.divider()


# =========================================================
# SİSTEM DURUMU
# =========================================================

with st.sidebar:

    st.header("⚙️ Sistem")

    try:

        client, model_name, foundry_url = (
            get_foundry_client()
        )

        st.success("Foundry Local: Hazır")

        st.write(
            f"**Model:** {model_name}"
        )

        st.write(
            f"**Sunucu:** {foundry_url}"
        )

    except Exception as e:

        st.error(
            "Foundry Local bağlantısı kurulamadı."
        )

        st.code(str(e))

        st.stop()


    try:

        vector_db = get_vector_db()

        st.success("Chroma: Hazır")

    except Exception as e:

        st.error(
            "Chroma veritabanı bulunamadı."
        )

        st.code(str(e))

        st.stop()


    st.divider()

    st.info(
        "Bu uygulama yerel çalışır. "
        "PDF'ler ve yapay zeka modeli "
        "yerel sistemden kullanılır."
    )


# =========================================================
# SORU AL
# =========================================================

query = st.text_area(
    "❓ Sorunuzu yazın",
    placeholder=(
        "Örneğin: "
        "Kalp krizi geçiren kişiye ne yapılmalı?"
    ),
    height=100
)


ask = st.button(
    "🔍 Cevapla",
    type="primary",
    use_container_width=True
)


# =========================================================
# CEVAP
# =========================================================

if ask:

    if not query.strip():

        st.warning(
            "Lütfen bir soru yazın."
        )

        st.stop()


    # -----------------------------------------------------
    # KONU
    # -----------------------------------------------------

    topic = detect_topic(query)

    st.write(
        f"🎯 **Tespit edilen konu:** "
        f"{topic if topic else 'Belirsiz'}"
    )


    # -----------------------------------------------------
    # KONU BULUNAMADI
    # -----------------------------------------------------

    if topic is None:

        st.warning(
            "Bu soru Acil Yardım Rehberi'nin "
            "kapsamı dışında veya konu belirlenemedi."
        )

        st.info(
            """
Sistem yalnızca PDF'lerde bulunan
ilk yardım konularına cevap verir.

Örnekler:

• Kalp krizi için ilk yardım
• Havale geçirene ne yapılır?
• Kanama nasıl kontrol edilir?
• Kene yapıştı ne yapmalıyım?
• Yanıkta ilk yardım nasıl yapılır?
"""
        )

        st.stop()


    # -----------------------------------------------------
    # PDF ARAMA
    # -----------------------------------------------------

    target_files = TOPIC_FILES.get(
        topic,
        []
    )

    st.write("🔍 **Dokümanlar taranıyor...**")

    st.write("📌 **Hedef dokümanlar:**")

    for file in target_files:

        st.write(
            f"- `{file}`"
        )


    docs = search_documents(
        vector_db,
        query,
        topic
    )


    # -----------------------------------------------------
    # DOKÜMAN BULUNAMADI
    # -----------------------------------------------------

    if not docs:

        st.warning(
            "Bu soru için uygun PDF içeriği bulunamadı."
        )

        st.info(
            "Modelin PDF dışından bilgi üretmesine "
            "izin verilmedi."
        )

        st.stop()


    context = create_context(docs)


    if not context.strip():

        st.warning(
            "PDF'den kullanılabilir içerik alınamadı."
        )

        st.stop()


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    with st.spinner(
        "🧠 Yapay zeka cevabı oluşturuyor..."
    ):

        try:

            answer = generate_answer(
                client,
                model_name,
                query,
                context
            )

        except Exception as e:

            st.error(
                "Foundry Local cevap hatası."
            )

            st.code(str(e))

            st.stop()


    # -----------------------------------------------------
    # CEVAP
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 Acil Yardım Asistanı"
    )

    st.write(answer)


    # -----------------------------------------------------
    # KAYNAKLAR
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "📚 Kullanılan kaynaklar"
    )

    shown_sources = set()

    for doc in docs:

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Bilinmeyen kaynak"
            )
        )

        if source not in shown_sources:

            st.write(
                f"- `{source}`"
            )

            shown_sources.add(source)