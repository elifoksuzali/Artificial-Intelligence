Dosya yapısı
Length Name
------ ----
    agent_ollama    --> agent dizayn edildiği yer
    database_ollama --> veri tabanındaki gerekli bilgileri çektiğimiz yer
    prompt_ollama   --> promptların çekildiği yer
    qdrant_ollama   --> qdrant bağlantı filtering/multivector işlemlerinin olduğu yer
    router_ollama   --> gelen mesajları gerekli agentlara gönderip işlemleri kontrol eden ana sınıf
    runner_ollama   --> agent run_sync ile çalıştırıldığı yer
    tool_ollama     --> farklı servislerden veri göndeirp alındığı yer
    api.py          --> post edilen verilerin alııp gerekli yerlere dağıtıldığı yer
    README.md       --> Proje açıklama klasörü
    setting.py      --> Gerekli local ayarların olduğu yer
    .env            --> setting.py dosyası için gerekli ana bilgilerin yer