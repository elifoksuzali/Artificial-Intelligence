<?php
function extracted(string $apiKey, CurlHandle|bool $ch): array
{
    $header = array(
        "Authorization: Bearer " . $apiKey,
        "Content-Type: application/json",
    );
    curl_setopt($ch, CURLOPT_HTTPHEADER, $header);
    $response = curl_exec($ch);
    if (curl_error($ch)) {
        echo 'Curl hatası: ' . curl_error($ch);
    }
    curl_close($ch);
    $response = json_decode($response, true);
    $message = $response['choices'][0]['message']['content'];
    $usage = $response['usage'];
    return compact('message','usage');
}
function formatResponse(string $message): string
{
    // Her başlığı yeni bir paragraf olarak düzenle
    $formattedMessage = preg_replace('/(\d+\.\s)/', "\n**$0**", $message);
    return $formattedMessage;
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['textareas'])) {
    $ch = curl_init();
    $textAreas = array_map('htmlspecialchars',$_POST['textareas']);

    $api_url = "https://api.openai.com/v1/chat/completions";

    curl_setopt($ch, CURLOPT_URL, $api_url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, 1);

    $dosyaYolu = $_SERVER["DOCUMENT_ROOT"] . "/demo/RunnerCode/api.txt";
    $apiKey = file_get_contents($dosyaYolu);

    if ($apiKey === false) {
        die("OpenAI Anahtar düzgün okunamadı");
    }

    $system_message = "Bir tur içerik üreticisi olarak hareket etmeni istiyorum. Ben sana tur içeriği hakkındaki bazı bilgileri yazacağım ve
       sen de bu bilgiler ile bir tur planı oluşturacaksın ve gezilecek olan yerin tarihi,önemli noktalarını betimleyerek anlatacaksın aynı zamanda
       gün gün yapılacak etkinlikleri belirtecek bu tur içeriğine özel bir tur başlığı ile beraber tur içerik oluşturacaksın. Bazı durumlarda sana 
       ziyaret edilecek yerleri,tur dahil hizmetleri,uçuş bilgileri,restoran bilgileri,tura dahil olan hizmetleri,tura dahil olmayana hizmetleri,
       konaklamaya, tur başlangıcı için toplanma yerlerini,turun son gününü belirterek içeriği oluşturacak son gün aktivitelerinden sonra 
       dönüşe geçiyor veya başka turlarda görüşmek dileğiyle gibi temenni cümlelerine yer vereceksin. Ayrıca bir içerik değil birden farklı 
       türde içerik oluşturup önereceksiniz tur içeriğinde temel unsurlar verildiğinde bunları betimleyerek içeriği oluştur. Gezilecek mekanların
        bilgisi,şehrin tarihi,doğası,turistlik açıdan önemi ve önemli noktalarını betimleyerek içeriği zengin bir tur metni üretmeni istiyorum.";

    $data = array(
        'model' => 'gpt-3.5-turbo-16k-0613',
        'messages' => [
            ["role" => "system", "content" => $system_message],
        ],
        "temperature" => 0.9,
        'max_tokens' => 4096,
        "top_p" => 1,
        "frequency_penalty" => 1,
        "presence_penalty" => 1,
    );
    foreach ($textAreas as $index => $textAreaContent) {
        $data['messages'][] = ["role" => "user", "content" => $textAreaContent];
    }


    $datas = json_encode($data);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $datas);
    set_time_limit(300);
    $result = extracted($apiKey, $ch);

    // Extract token information
    $promptTokens = $result['usage']['prompt_tokens'];
    $completionTokens = $result['usage']['completion_tokens'];
    $totalTokens = $result['usage']['total_tokens'];

    // Open ai den geide-gelen cevabın maliyetini hesaplama
    $costPerToken = 0.0030; // openai model fiyat ($0.0030 / 1K input tokens + $0.0040 / 1K output tokens)
    $costPerOutputToken = 0.0040; // Çıkış token'ları için maliyet

    $cost = ($totalTokens * $costPerToken + $totalTokens * $costPerOutputToken) / 1000;

    // dönen cevabın formatını düzenliyoruz
    $formattedMessage = formatResponse($result['message']);
    $results = nl2br($formattedMessage); // Satır sonları için HTML line breaks ekleyerek görüntüle

    echo "Cevap:", $results . '<br>';
    echo "Token Sayısı:", $totalTokens . '<br>';
    echo "Maliyet (tahmini): $" . number_format($cost, 2) . '<br>';
    echo "Prompt Token Sayısı(100 tokens ~= 75 words): " . $promptTokens . PHP_EOL . '<br>';
    echo "Completion Token Sayısı: " . $completionTokens . PHP_EOL . '<br>';
    echo "Toplam Token Sayısı: " . $totalTokens . PHP_EOL . '<br>';
}

?>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Tatil Güzergahı Oluşturucu</title>
</head>
<body>
<h1>Seyahat Programı</h1>
<form method="post">
    <div class="textarea-container">

        <textarea name="textareas[]" placeholder="Tur Kalkış Noktaları" required></textarea><br>
        <textarea name="textareas[]" placeholder="Tur Başlık" required cols="30" rows="5"></textarea><br>
        <textarea name="textareas[]" placeholder="Tur Tipi (Gemi - Uçak - Otobüs - Araç Kiralamaları)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Tur Başlangıç Noktası (Havalimanı)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Gün:(Tur kaç gün sürecek ve dönüş gününü belirterek yazınız)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Şehir/Bölge:(Tur düzenlendiği şehir/ler)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Ziyaret Edilecek Yerler (alanlar,ören yerleri)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Tur Teması (Doğa-Tarihi vb.)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Konaklama yapılacak alan (otel,ev,villa vb.)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Restoranlar (cafe,restoran vb.)" required></textarea><br>
        <textarea name="textareas[]" placeholder="Fiyata Dahil Hizmetler" required></textarea><br>
        <textarea name="textareas[]" placeholder="Fiyata Dahil Olmayan Hizmetler" required></textarea><br>
        <input type="submit" name="submit" value="Enter" style="margin: 100px 600px; width: 100px;height: 50px;">

    </div>
</form>

</body>
</html>


<style>
    .textarea-container {
        display: flex;
        margin-bottom: 10px; /* Adjust the margin to your preference */
        justify-content: space-around;
        flex-direction: row-reverse;
        align-items: flex-end;
        align-content: space-around;
        flex-wrap: wrap;
    }

    textarea {
        width: 500px;
        height: 100px;
        margin-right: 10px;
        margin-bottom: 10px;
    }

</style>


