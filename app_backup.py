import os

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# =========================================================
# AYARLAR
# =========================================================

DB_PATH = "./chroma_db"

# ŞU ANKİ FOUNDY PORTUN
FOUNDRY_URL = "http://127.0.0.1:59128/v1"

# Foundry'de yüklediğin model
FOUNDRY_MODEL_NAME = "phi-3.5-mini"


# =========================================================
# KONU - PDF EŞLEŞTİRMESİ
# =========================================================

TOPIC_DOCUMENTS = {

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

    "bayilma": [
        "bayilma_durumu.pdf"
    ],

    "diyabet": [
        "diyabet_ilk_yardim.pdf"
    ],

    "kedi_kopek_isirmasi": [
        "kedi_kopek_isirmasi_ilk_yardim.pdf"
    ],

    "kene": [
        "kene.pdf"
    ],

    "kirik": [
        "kirik_durumu.pdf"
    ],

    "sara_epilepsi": [
        "sara_epilepsi_ilk_yardim.pdf"
    ],

    "sarilik": [
        "sarilik_ilk_yardim.pdf"
    ],

    "yanik": [
        "yanma_durumu.pdf"
    ],

    "ari_akrep_yilan": [
        "ari_akrep_yilan_sokmasi_ilk_yardim.pdf"
    ],

    "yuksek_ates_havalesi": [
        "yuksek_ates_havalesi_ilk_yardim.pdf"
    ],

    "genel_ilk_yardim": [
        "Ilk_Yardim_Rehberi.pdf"
    ]
}


# =========================================================
# KONU TESPİTİ
# =========================================================

def detect_topic(query):

    q = query.lower().strip()

    # -----------------------------------------------------
    # KALP KRİZİ
    # -----------------------------------------------------

    if any(word in q for word in [
        "kalp krizi",
        "kalp krizi geçiriyor",
        "kalp krizi geçiriyor",
        "göğüs ağrısı",
        "gogus agrisi",
        "göğsü ağrıyor",
        "gogsu ağriyor",
        "göğsüm ağrıyor",
        "gogsum ağriyor",
        "kalbim ağrıyor",
        "kalbim sıkışıyor",
        "kalp sıkışması"
    ]):
        return "kalp_krizi"


    # -----------------------------------------------------
    # HAVALE
    # -----------------------------------------------------

    if any(word in q for word in [
        "havale",
        "havale geçiriyor",
        "havale geçirme",
        "ateşli havale",
        "ateş havalesi",
        "atesli havale",
        "ates havalesi"
    ]):

        if any(word in q for word in [
            "ateş",
            "ates",
            "ateşli",
            "atesli"
        ]):
            return "yuksek_ates_havalesi"

        return "havale"


    # -----------------------------------------------------
    # KANAMA
    # -----------------------------------------------------

    if any(word in q for word in [
        "kanama",
        "kanıyor",
        "kan kaybı",
        "çok kanıyor",
        "cok kaniyor"
    ]):
        return "kanama"


    # -----------------------------------------------------
    # BAYILMA
    # -----------------------------------------------------

    if any(word in q for word in [
        "bayılma",
        "bayıldı",
        "bayildi",
        "bayılıyor",
        "bayiliyor",
        "bilinci kapalı",
        "bilinci kapandi"
    ]):
        return "bayilma"


    # -----------------------------------------------------
    # DİYABET
    # -----------------------------------------------------

    if any(word in q for word in [
        "diyabet",
        "şeker hastası",
        "seker hastasi",
        "kan şekeri",
        "kan sekeri",
        "şekeri düştü",
        "sekeri düştü",
        "şekerim düştü"
    ]):
        return "diyabet"


    # -----------------------------------------------------
    # KEDİ / KÖPEK
    # -----------------------------------------------------

    if any(word in q for word in [
        "köpek ısırdı",
        "kopek isirdi",
        "köpek ısırması",
        "kopek isirmasi",
        "kedi ısırdı",
        "kedi isirdi",
        "kedi tırmaladı",
        "kedi tirmaladi",
        "hayvan ısırması"
    ]):
        return "kedi_kopek_isirmasi"


    # -----------------------------------------------------
    # KENE
    # -----------------------------------------------------

    if any(word in q for word in [
        "kene",
        "kene yapıştı",
        "kene yapisti",
        "kene ısırdı",
        "kene isirdi"
    ]):
        return "kene"


    # -----------------------------------------------------
    # KIRIK
    # -----------------------------------------------------

    if any(word in q for word in [
        "kırık",
        "kirik",
        "kemiği kırıldı",
        "kemigi kirildi",
        "kolu kırıldı",
        "kolu kirildi",
        "bacağı kırıldı",
        "bacagi kirildi"
    ]):
        return "kirik"


    # -----------------------------------------------------
    # SARA / EPİLEPSİ
    # -----------------------------------------------------

    if any(word in q for word in [
        "sara",
        "epilepsi",
        "epilepsi nöbeti",
        "epilepsi nobeti",
        "sara nöbeti",
        "sara nobeti",
        "nöbet geçiriyor",
        "nobet geciriyor"
    ]):
        return "sara_epilepsi"


    # -----------------------------------------------------
    # SARILIK
    # -----------------------------------------------------

    if any(word in q for word in [
        "sarılık",
        "sarilik",
        "sarardı",
        "sarardi",
        "gözleri sarı",
        "gozleri sari"
    ]):
        return "sarilik"


    # -----------------------------------------------------
    # YANIK
    # -----------------------------------------------------

    if any(word in q for word in [
        "yanık",
        "yanik",
        "yandı",
        "yandi",
        "sıcak su döküldü",
        "sicak su dokuldu",
        "kaynar su",
        "ateş yaktı"
    ]):
        return "yanik"


    # -----------------------------------------------------
    # ARI / AKREP / YILAN
    # -----------------------------------------------------

    if any(word in q for word in [
        "arı soktu",
        "ari soktu",
        "arı sokması",
        "ari sokmasi",
        "akrep soktu",
        "akrep sokması",
        "akrep sokmasi",
        "yılan soktu",
        "yilan soktu",
        "yılan ısırdı",
        "yilan isirdi",
        "yılan sokması",
        "yilan sokmasi"
    ]):
        return "ari_akrep_yilan"


    # -----------------------------------------------------
    # GENEL İLK YARDIM
    # -----------------------------------------------------

    if any(word in q for word in [
        "ilk yardım",
        "ilk yardim",
        "acil yardım",
        "acil yardim",
        "ilk yardım nedir",
        "ilk yardim nedir"
    ]):
        return "genel_ilk_yardim"


    return "Belirsiz"


# =========================================================
# FOUNDRY LOCAL
# =========================================================

def get_foundry_client():

    try:

        client = OpenAI(
            base_url=FOUNDRY_URL,
            api_key="foundry"
        )

        models = client.models.list()

        print("✅ Foundry Local bağlantısı başarılı.")
        print()
        print("📦 Foundry modelleri:")

        model_names = []

        for model in models.data:

            print(f"- {model.id}")
            model_names.append(model.id)

        print()

        selected_model = None

        for model_id in model_names:

            if "phi-3.5-mini" in model_id.lower():

                selected_model = model_id
                break


        if selected_model is None:

            print("❌ Phi-3.5-mini modeli bulunamadı.")
            print()

            for model_id in model_names:
                print(f"   - {model_id}")

            return None, None


        print("✅ Kullanılacak Foundry modeli:")
        print(f"   {selected_model}")

        return client, selected_model


    except Exception as e:

        print("❌ Foundry Local bağlantı hatası:")
        print(e)

        return None, None


# =========================================================
# SADECE HEDEF PDF'LERDEN CHUNKLARI AL
# =========================================================

def get_documents_from_sources(vector_db, target_documents):

    all_docs = []

    try:

        for pdf_name in target_documents:

            result = vector_db.get(
                where={
                    "source": pdf_name
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])

            for i, text in enumerate(documents):

                if not text:
                    continue

                metadata = {}

                if i < len(metadatas):
                    metadata = metadatas[i] or {}

                all_docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )


        # Chunk sırasını koru
        all_docs.sort(
            key=lambda doc: (
                doc.metadata.get("source", ""),
                doc.metadata.get("chunk_id", 0)
            )
        )

        return all_docs


    except Exception as e:

        print("❌ PDF içerikleri alınırken hata oluştu:")
        print(e)

        return []


# =========================================================
# ANA PROGRAM
# =========================================================

def main():

    print()
    print("🔄 Acil Yardım RAG sistemi başlatılıyor...")
    print()


    # =====================================================
    # 1. EMBEDDING
    # =====================================================

    print("🔢 Türkçe embedding modeli yükleniyor...")

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        )

        print("✅ Türkçe embedding modeli hazır.")

    except Exception as e:

        print("❌ Embedding modeli yüklenemedi:")
        print(e)

        return


    print()


    # =====================================================
    # 2. CHROMA
    # =====================================================

    if not os.path.exists(DB_PATH):

        print("❌ chroma_db klasörü bulunamadı.")
        print()
        print("Önce:")
        print("python ingest.py")

        return


    print("📚 Chroma veritabanı açılıyor...")

    try:

        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

        print("✅ Chroma hazır.")

    except Exception as e:

        print("❌ Chroma başlatılamadı:")
        print(e)

        return


    print()


    # =====================================================
    # 3. FOUNDRY
    # =====================================================

    print("🤖 Foundry Local bağlantısı kuruluyor...")

    client, model_name = get_foundry_client()

    if client is None:

        print()
        print("⚠️ Foundry Local'a bağlanılamadı.")

        return


    print()
    print("=" * 55)
    print("🚑 ACİL YARDIM ASİSTANI HAZIR")
    print("=" * 55)
    print()
    print(f"🧠 Model: {model_name}")
    print()
    print("Çıkmak için q yaz.")
    print()


    # =====================================================
    # 4. SORU - CEVAP
    # =====================================================

    while True:

        print()

        query = input("❓ Sorunuz: ").strip()


        # -------------------------------------------------
        # ÇIKIŞ
        # -------------------------------------------------

        if query.lower() == "q":

            print()
            print("Görüşmek üzere.")

            break


        if not query:
            continue


        # =================================================
        # 5. KONU TESPİTİ
        # =================================================

        topic = detect_topic(query)

        print()
        print(f"🎯 Tespit edilen konu: {topic}")


        # =================================================
        # 6. BELİRSİZ SORU
        # =================================================

        if topic == "Belirsiz":

            print()
            print(
                "⚠️ Bu soru Acil Yardım Rehberi'nin "
                "kapsamı dışında veya konu belirlenemedi."
            )

            print()
            print(
                "📚 Sistem yalnızca PDF'lerde bulunan "
                "ilk yardım konularına cevap verir."
            )

            print()
            print("Örnek sorular:")

            print("   • Kalp krizi için ilk yardım")
            print("   • Havale geçirene ne yapılır?")
            print("   • Kanama nasıl kontrol edilir?")
            print("   • Kene yapıştı ne yapmalıyım?")
            print("   • Yanıkta ilk yardım nasıl yapılır?")

            continue


        # =================================================
        # 7. HEDEF PDF'LER
        # =================================================

        target_documents = TOPIC_DOCUMENTS.get(
            topic,
            []
        )


        if not target_documents:

            print()
            print("⚠️ Bu konu için PDF tanımlanmamış.")

            continue


        print()
        print("🔍 Dokümanlar taranıyor...")

        print()
        print("📌 Hedef dokümanlar:")

        for filename in target_documents:

            print(f"   - {filename}")


        # =================================================
        # 8. PDF İÇERİKLERİNİ AL
        # =================================================

        docs = get_documents_from_sources(
            vector_db,
            target_documents
        )


        if not docs:

            print()
            print(
                "⚠️ Hedef PDF'lerde içerik bulunamadı."
            )

            print()
            print(
                "⚠️ Modelin alakasız cevap üretmesi engellendi."
            )

            continue


        # =================================================
        # 9. CONTEXT
        # =================================================

        context_parts = []

        used_sources = []

        for doc in docs:

            text = doc.page_content.strip()

            if len(text) < 20:
                continue


            source = doc.metadata.get(
                "source",
                "Bilinmeyen kaynak"
            )

            context_parts.append(
                f"[{source}]\n{text}"
            )


            if source not in used_sources:

                used_sources.append(source)


        if not context_parts:

            print()
            print(
                "⚠️ Kullanılabilir PDF içeriği bulunamadı."
            )

            continue


        context = "\n\n".join(context_parts)


        # =================================================
        # 10. PROMPT
        # =================================================

        prompt = f"""
Sen Acil Yardım Rehberi adlı yerel RAG sisteminin
ilk yardım asistanısın.

KULLANICININ SORUSU:
{query}

KONU:
{topic}

AŞAĞIDAKİ METİNLER SADECE İLGİLİ PDF'LERDEN
ALINMIŞTIR:

{context}


GÖREV:

Kullanıcının sorusuna yalnızca yukarıdaki PDF
içeriğinde bulunan bilgilere dayanarak cevap ver.


KURALLAR:

- PDF'de bulunmayan hiçbir bilgiyi ekleme.
- Genel tıbbi bilginden cevap üretme.
- Tahmin yapma.
- PDF'deki bilgileri değiştirme.
- PDF içeriğinde olmayan bir adımı kendin oluşturma.
- Kullanıcıya doğrudan cevap ver.
- "PDF'ye göre", "kaynaklara göre",
  "PDF metinlerinden alınan bilgiler" gibi ifadeler kullanma.
- Teknik RAG açıklaması yapma.
- "Konu:" veya "Kaynak:" şeklinde cevap verme.
- Cevabı Türkçe yaz.
- Gereksiz uzun açıklamalar yapma.
- PDF'de birden fazla işlem/adım varsa bunları
  numaralı şekilde sırala.
- PDF'de yalnızca birkaç bilgi varsa yalnızca
  o bilgileri kullan.
- PDF'de cevap bulunmuyorsa:
  "Verilen PDF'lerde bu sorunun cevabı bulunmuyor."
  şeklinde belirt.
- Tıbbi teşhis koyma.


CEVAP FORMATI:

Gerekirse tek cümlelik kısa bir açıklama.

1. İlk adım...
2. İkinci adım...
3. Üçüncü adım...


SADECE KULLANICIYA VERİLECEK CEVABI YAZ.
"""


        # =================================================
        # 11. FOUNDRY
        # =================================================

        print()
        print("🧠 Yapay zeka cevabı oluşturuyor...")


        try:

            response = client.chat.completions.create(

                model=model_name,

                messages=[

                    {
                        "role": "system",
                        "content": (
                            "Sen yalnızca verilen PDF "
                            "içeriğine dayanan Türkçe "
                            "ilk yardım asistanısın. "
                            "PDF'de bulunmayan bilgi "
                            "üretme."
                        )
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                temperature=0.0,

                max_tokens=500
            )


            answer = response.choices[0].message.content


            # =================================================
            # 12. CEVAP
            # =================================================

            print()
            print("🤖 ACİL YARDIM ASİSTANI")
            print("-" * 55)
            print(answer.strip())
            print("-" * 55)


            # =================================================
            # 13. KAYNAKLAR
            # =================================================

            print()
            print("📚 Kullanılan kaynaklar:")

            for i, source in enumerate(
                used_sources,
                start=1
            ):

                print(
                    f"   [{i}] {source}"
                )


        except Exception as e:

            print()
            print("❌ Foundry Local cevap hatası:")
            print(e)


# =========================================================
# PROGRAMI BAŞLAT
# =========================================================

if __name__ == "__main__":
    main()