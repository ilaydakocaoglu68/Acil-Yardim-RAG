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


# =========================================================
# STREAMLIT SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Acil Yardım Rehberi",
    page_icon="🚑",
    layout="centered"
)


# =========================================================
# BAŞLIK
# =========================================================

st.title("🚑 Acil Yardım Rehberi")

st.caption(
    "Yerel RAG • Foundry Local • Chroma"
)

st.info(
    "Bu uygulama yalnızca yerel PDF kaynaklarındaki "
    "ilk yardım bilgilerini kullanır."
)


# =========================================================
# FOUNDRY SERVER ADRESİNİ BUL
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
            r"Web URLs\s+(http://127\.0\.0\.1:\d+)",
            output
        )

        if match:
            return match.group(1) + "/v1"

        return None

    except Exception:
        return None


# =========================================================
# FOUNDRY CLIENT
# =========================================================

@st.cache_resource
def get_foundry_client():

    foundry_url = get_foundry_url()

    if not foundry_url:
        return None, None

    try:

        client = OpenAI(
            base_url=foundry_url,
            api_key="foundry"
        )

        models = client.models.list()

        selected_model = None

        for model in models.data:

            model_id = model.id

            if "phi-3.5-mini" in model_id.lower():
                selected_model = model_id
                break

        if selected_model is None:
            return None, None

        return client, selected_model

    except Exception:
        return None, None


# =========================================================
# EMBEDDING + CHROMA
# =========================================================

@st.cache_resource
def load_database():

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vector_db


# =========================================================
# KONU TESPİTİ
# =========================================================

TOPICS = {

    "kalp_krizi": [
        "kalp krizi",
        "kalp krizi geçiriyor",
        "kalp krizi geçiren",
        "göğüs ağrısı",
        "kalp"
    ],

    "havale": [
        "havale",
        "havale geçiriyor",
        "havale geçiren",
        "nöbet",
        "ateşli havale"
    ],

    "kanama": [
        "kanama",
        "kanıyor",
        "kanayan",
        "çok kanıyor"
    ],

    "kene": [
        "kene",
        "kene yapıştı",
        "kene ısırdı"
    ],

    "yanma": [
        "yanık",
        "yanma",
        "yandı",
        "sıcak su döküldü"
    ],

    "bayilma": [
        "bayılma",
        "bayıldı",
        "baygın"
    ],

    "kırık": [
        "kırık",
        "kemiği kırıldı",
        "kolu kırıldı",
        "bacağı kırıldı"
    ],

    "kedi_kopek_isirmasi": [
        "köpek ısırdı",
        "kedi ısırdı",
        "köpek ısırması",
        "kedi ısırması",
        "hayvan ısırdı"
    ],

    "sara_epilepsi": [
        "epilepsi",
        "sara",
        "epilepsi nöbeti",
        "sara nöbeti"
    ],

    "diyabet": [
        "diyabet",
        "şeker hastası",
        "şekeri düştü",
        "kan şekeri"
    ],

    "arı_akrep_yılan": [
        "arı soktu",
        "arı sokması",
        "akrep soktu",
        "akrep sokması",
        "yılan soktu",
        "yılan ısırdı"
    ],

    "sarılık": [
        "sarılık",
        "sarardı",
        "cilt sarardı"
    ],

    "yüksek_ateş": [
        "yüksek ateş",
        "ateşi çok yüksek",
        "ateş"
    ]
}


TOPIC_FILES = {

    "kalp_krizi": [
        "kalp_krizi.pdf"
    ],

    "havale": [
        "havale_genel_ilk_yardim.pdf",
        "yuksek_ates_havalesi_ilk_yardim.pdf"
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

    "bayilma": [
        "bayilma_durumu.pdf"
    ],

    "kırık": [
        "kirik_durumu.pdf"
    ],

    "kedi_kopek_isirmasi": [
        "kedi_kopek_isirmasi_ilk_yardim.pdf"
    ],

    "sara_epilepsi": [
        "sara_epilepsi_ilk_yardim.pdf"
    ],

    "diyabet": [
        "diyabet_ilk_yardim.pdf"
    ],

    "arı_akrep_yılan": [
        "ari_akrep_yilan_sokmasi_ilk_yardim.pdf"
    ],

    "sarılık": [
        "sarilik_ilk_yardim.pdf"
    ],

    "yüksek_ateş": [
        "yuksek_ates_havalesi_ilk_yardim.pdf"
    ]
}


# =========================================================
# KONU BUL
# =========================================================

def detect_topic(query):

    query_lower = query.lower()

    for topic, keywords in TOPICS.items():

        for keyword in keywords:

            if keyword in query_lower:
                return topic

    return None


# =========================================================
# DOKÜMAN ARAMA
# =========================================================

def search_documents(vector_db, query, topic):

    # Önce geniş arama
    results = vector_db.similarity_search(
        query,
        k=3
    )

    if not results:
        return []

    # Konu belirlendiyse yalnızca o konuya ait PDF'leri kullan
    if topic and topic in TOPIC_FILES:

        target_files = set(
            TOPIC_FILES[topic]
        )

        filtered = []

        for doc in results:

            source = os.path.basename(
                doc.metadata.get(
                    "source",
                    ""
                )
            )

            if source in target_files:
                filtered.append(doc)

        return filtered

    return results


# =========================================================
# CONTEXT OLUŞTUR
# =========================================================

def build_context(docs):

    parts = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        text = doc.page_content.strip()

        if len(text) < 20:
            continue

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Bilinmeyen kaynak"
            )
        )

        parts.append(
            f"[Kaynak {i}: {source}]\n{text}"
        )

    return "\n\n".join(parts)


# =========================================================
# CEVAP ÜRET
# =========================================================

def generate_answer(
    client,
    model_name,
    query,
    context
):

    prompt = f"""
Sen Acil Yardım Rehberi adlı yerel RAG
sisteminin ilk yardım asistanısın.

ÇOK ÖNEMLİ:

- Yalnızca verilen PDF içeriğine dayan.
- PDF'de olmayan bilgiyi ekleme.
- Bilgi PDF'de yoksa bunu açıkça söyle.
- Türkçe cevap ver.
- Kısa ve anlaşılır cevap ver.
- En fazla 5 madde kullan.
- Gereksiz açıklama yapma.
- Tıbbi teşhis koyma.
- Kaynakta belirtiliyorsa 112 bilgisini aktar.

KULLANICI SORUSU:
{query}

PDF KAYNAĞI:
{context}

Sadece yukarıdaki PDF içeriğine dayanarak cevap ver.
"""

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "system",
                "content": (
                    "Sen yalnızca verilen PDF "
                    "kaynaklarına dayanan Türkçe "
                    "ilk yardım asistanısın."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_tokens=180
    )

    return response.choices[0].message.content


# =========================================================
# SİSTEMLERİ YÜKLE
# =========================================================

try:

    vector_db = load_database()

except Exception as e:

    st.error(
        "❌ Chroma veritabanı yüklenemedi."
    )

    st.code(str(e))

    st.stop()


client, model_name = get_foundry_client()


if client is None:

    st.error(
        "❌ Foundry Local'a bağlanılamadı."
    )

    st.warning(
        "Terminalde `foundry server status` "
        "komutuyla servisin Ready olduğundan emin ol."
    )

    st.stop()


# =========================================================
# ARAYÜZ
# =========================================================

st.success(
    f"🟢 Sistem hazır • Model: {model_name}"
)

st.divider()

query = st.text_area(
    "❓ Sorunuzu yazın",
    placeholder=(
        "Örneğin: Kalp krizi geçirene "
        "ne yapılır?"
    ),
    height=100
)


answer_button = st.button(
    "🔍 CEVAPLA",
    type="primary",
    use_container_width=True
)


# =========================================================
# CEVAP
# =========================================================

if answer_button:

    if not query.strip():

        st.warning(
            "Lütfen bir soru yazın."
        )

        st.stop()

    topic = detect_topic(query)

    if topic is None:

        st.warning(
            "⚠️ Bu soru Acil Yardım Rehberi'nin "
            "kapsamı dışında veya konu belirlenemedi."
        )

        st.info(
            "Sistem yalnızca PDF'lerde bulunan "
            "ilk yardım konularına cevap verir."
        )

        st.stop()

    st.write(
        f"🎯 **Tespit edilen konu:** `{topic}`"
    )

    with st.spinner(
        "🔍 Dokümanlar taranıyor..."
    ):

        docs = search_documents(
            vector_db,
            query,
            topic
        )

    if not docs:

        st.warning(
            "⚠️ Bu soru için uygun PDF içeriği bulunamadı."
        )

        st.stop()

    context = build_context(docs)

    if not context:

        st.warning(
            "⚠️ Kullanılabilir PDF içeriği bulunamadı."
        )

        st.stop()

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
                "❌ Cevap oluşturulurken hata oluştu."
            )

            st.code(str(e))

            st.stop()

    st.divider()

    st.subheader(
        "🤖 Acil Yardım Asistanı"
    )

    st.write(answer)

    st.divider()

    st.subheader(
        "📚 Kullanılan kaynaklar"
    )

    used_sources = []

    for doc in docs:

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Bilinmeyen kaynak"
            )
        )

        if source not in used_sources:
            used_sources.append(source)

    for source in used_sources:

        st.write(
            f"📄 `{source}`"
        )