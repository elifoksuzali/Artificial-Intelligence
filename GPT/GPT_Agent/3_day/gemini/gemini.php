<?php
$flask_url = 'http://127.0.0.1:5000/gemini'; // Flask uygulamasının adresini güncelleyin

$prediction = "";

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Formdan gelen metin verisini alın
    $text_data = isset($_POST['text_input']) ? $_POST['text_input'] : '';

    // JSON formatında POST verileri oluştur
    $post_fields = json_encode(['text' => $text_data]);

    // cURL başlat
    $ch = curl_init();

    // Flask API'sine POST isteği gönderme ayarları
    curl_setopt($ch, CURLOPT_URL, $flask_url);
    curl_setopt($ch, CURLOPT_POST, 1);

    // POST verilerini JSON formatında gönder
    curl_setopt($ch, CURLOPT_POSTFIELDS, $post_fields);

    // İçerik türünü "application/json" olarak ayarla
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);

    // Curl'dan yanıt almak için
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    // Yanıtı al
    $response = curl_exec($ch);

    if ($response === false) {
        $prediction = "Error: " . curl_error($ch); // Hataları göster
    } else {
        // JSON yanıtını diziye dönüştür
        $json_response = json_decode($response, true);

        // Tur içeriğini al
        if (isset($json_response['tur'])) {
            // Yeni satırları HTML'de göstermek için str_replace ile <br> ekleyin
            $tur_content = nl2br(htmlspecialchars($json_response['tur']));
            $prediction = "Tur İçeriği:<br>" . $tur_content;
        } else {
            $prediction = "No content found in response.";
        }
    }

    // cURL oturumunu kapat
    curl_close($ch);
}
?>

<html>
<head>
    <title>Flask API Tur İçeriği</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<h1>Tur İçeriği ile Flask API</h1>
<form action="/" method="post">
    <!-- Metin alanı -->
    <textarea name="text_input" rows="4" cols="50" placeholder="Metni girin..."></textarea><br>
    <input type="submit" value="Tahmin Et">
</form>
<!-- Yanıtı görüntüle -->
<p><?= $prediction ?></p>
</body>
</html>
