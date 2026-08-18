# 🚑 Acil Yardım RAG

ilk yardım dokümanları üzerinden çalışan, yerel yapay zekâ destekli bir
Retrieval-Augmented Generation (RAG) uygulamasıdır.

## 📌 Proje Hakkında

Bu uygulamanın amacı, ilk yardım konusunda kullanıcı tarafından sorulan sorulara,
proje içerisinde bulunan PDF dokümanlarından ilgili bilgileri getirerek cevap
vermektir.

Sistem, kullanıcının sorusunu analiz eder, soruyla ilgili ilk yardım konusunu
belirler ve ilgili PDF dokümanlarında arama gerçekleştirir.

Bulunan bilgiler daha sonra yerel olarak çalışan yapay zekâ modeli tarafından
kullanılarak kullanıcıya cevap oluşturulur.

Proje özellikle ilk yardım bilgilerinin belirli kaynak dokümanlara
dayandırılması ve konu dışındaki sorulara cevap verilmemesi üzerine
tasarlanmıştır.

---

## 🧠 RAG Sistemi Nasıl Çalışıyor?

Proje temel olarak şu adımlardan oluşmaktadır:

1. Kullanıcı ilk yardım ile ilgili bir soru sorar.
2. Sistem sorunun konusunu belirler.
3. İlgili PDF dokümanları seçilir.
4. PDF içerisindeki bilgiler Chroma veritabanında aranır.
5. Kullanıcı sorusuyla en alakalı bilgiler bulunur.
6. Bulunan bilgiler yapay zekâ modeline aktarılır.
7. Model, bulunan kaynak bilgilerine dayanarak cevap oluşturur.
8. Kullanılan PDF kaynakları kullanıcıya gösterilir.

### Genel Akış

```text
Kullanıcı Sorusu
       ↓
Konu Belirleme
       ↓
İlgili PDF'lerin Seçilmesi
       ↓
ChromaDB Araması
       ↓
İlgili Doküman İçerikleri
       ↓
Foundry Local
       ↓
Phi-3.5-mini
       ↓
Cevap + Kaynaklar

🌐 Yerel ve İnternetsiz Çalışma

Projenin önemli özelliklerinden biri yerel çalışacak şekilde tasarlanmış
olmasıdır.

Cevap üretiminde kullanılan yapay zekâ modeli Foundry Local üzerinden
çalıştırıldığı için sistemin temel RAG ve LLM işlemleri yerel bilgisayar
üzerinde gerçekleştirilebilir.

İlk kurulum ve model/doküman hazırlama işlemlerinden sonra uygulamanın
çalışması için sürekli bir internet bağlantısına ihtiyaç duyulmaması
hedeflenmiştir.


🚀 Kurulum

Projeyi bilgisayarınıza indirdikten sonra proje klasörüne geçin:

cd local-rag-app

Sanal ortamı aktif edin:

.\.venv\Scripts\Activate.ps1

Gerekli Python paketlerini yükleyin:

pip install -r gereksinimler.txt

Foundry Local üzerinde kullanılan modeli yükleyin:

foundry model load phi-3.5-mini

Ardından Streamlit uygulamasını başlatın:

streamlit run streamlit_app.py

Tarayıcı üzerinden Streamlit tarafından verilen yerel adrese
giderek uygulama kullanılabilir.
