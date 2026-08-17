import sys
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    print("🔄 Foundry Local yerel servisi yapılandırılıyor...")
    try:
        # 1. Uygulama adını vererek konfigürasyonu oluşturuyoruz
        config = Configuration(app_name="local-rag-app")
        
        # 2. Servis yöneticisini başlatıyoruz
        manager = FoundryLocalManager(config=config)
        
        print("✅ Foundry Local Manager başarıyla başlatıldı!")
        print(f"Yönetici Nesnesi: {manager}")
        print("\n🚀 Arka plan servisi aktif ve hazır! Yerel RAG altyapımıza geçebiliriz.")
        
    except Exception as e:
        print(f"❌ Servis başlatılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()