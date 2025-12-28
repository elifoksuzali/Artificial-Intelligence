
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Drawing.Imaging;
using System.Diagnostics;


namespace ImageProcessingProject
{
    public partial class AnaEkran : Form
    {
        private string selectedPath;
        private int CellWidth = 1;
        private int CellHeight = 1;
        private int BlockWidth = 1;
        private int BlockHeight = 1;
        private bool SignedOrientations = false;

        public AnaEkran()
        {
            InitializeComponent();
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            
            classifierComboBox.Items.Add("LBP");
            classifierComboBox.Items.Add("HOG");
        }
        //klasör içindeki dosyaalra eriştik ve içinden görselimiz seçtik
        private void uploadButton_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                DialogResult result = fbd.ShowDialog();

                if (result == DialogResult.OK && !string.IsNullOrWhiteSpace(fbd.SelectedPath))
                {
                    selectedPath = fbd.SelectedPath;
                    string[] files = Directory.GetFiles(fbd.SelectedPath);

                    var message = new StringBuilder();
                    message.AppendLine("Path: " + fbd.SelectedPath);
                    message.AppendLine("");
                    message.AppendLine("Files found: " + files.Length.ToString());

                    if (files.Length > 0)
                    {
                        var s = new StringBuilder();
                        s.AppendLine("");
                        var i = 1;
                        foreach (var item in files)
                        {
                            s.AppendLine(i + " - " + item.ToString());
                            i++;
                        }
                        message.AppendLine(s.ToString());
                    }

                    MessageBox.Show(message.ToString(), "Message");
                }
            }
        }

        // Local Binary Pattern için uygulanacak olan maske 

        //LBP'nin ana özellikleri şunlardır:

        //  1-Düşük hesaplama maliyeti

        //  2-Görüntü gri skala değerlerindeki dalgalanmalara karşı direnç

        // -------------------- ADIMLAR ---------------------------------------\\

        //Görüntüyü gri tonlamalı alana dönüştürün.
        //Görüntüdeki her piksel(gp) için merkezi pikseli çevreleyen P komşuluklarını seçin. gp koordinatları tarafından verilir


        Bitmap ObtainMask(Bitmap Src)
        {

            //. İşaret taraması için kullanacağımız yerel bir görüntü bmp'si alın
            Bitmap bmp = (Bitmap)Src.Clone();
            int NumRow = pictureBox1.Height;
            int numCol = pictureBox1.Width;
            //. Maske görüntüsü, başlangıçta aynı boyutta bir kaynak görüntüsüdür
            Bitmap mask = new Bitmap(pictureBox1.Width, pictureBox1.Height);// GRİ sonuç matrisidir
            ///3. Maskeyi genişletmek için bnd boyutunda bir yapılandırma elemanı tanımlayacağız.
            // Bu maske kalınlaştırma değişkenini kullanıcıdan alabilirsiniz. ancak 3, 256x256 görüntüler için standarttır
            int bnd = 3;
            ///4. Görüntülerin satırları ve sütunları arasında dolaşın
            for (int i = 0; i < NumRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {
                    Color c = bmp.GetPixel(j, i);// Bir pikselin rengini çıkar
                    int rd = c.R; int gr = c.G; int bl = c.B;// renkten kırmızı, yeşil, mavi bileşenleri ayıklayın.
                    // Bölgeyi kırmızı ile boyadığınızı unutmayın. Ancak görüntü yeniden boyutlandırıldı, bu yeniden örnekleniyor
                    // yeniden boyutlandırma işleminde renk değişme eğilimindedir. bu yüzden kırmızıdan daha fazla kırmızımsı piksel değerlendirici arayacağız.
                    // Maske rengini fare tıklamasıyla seçerek de bunu güncelleyebilirsiniz.
                    if ((rd > 220) && (gr < 80) && (bl < 80))
                    {
                        Color c2 = Color.FromArgb(255, 255, 255);
                        //5. Im'de beyaz olarak işaretlenen pikseli ayarla
                        mask.SetPixel(j, i, c2);

                        ///6. Genişletme gerçekleştirin (kenar efektini geçersiz kılmak için maske alanını genişletin
                        for (int ib = i - bnd; ib < i + bnd; ib++)
                        {
                            for (int jb = j - bnd; jb < j + bnd; jb++)
                            {
                                try
                                {
                                    //piksellerin sınırını da beyaz yapıyoruz
                                    mask.SetPixel(jb, ib, c2);
                                }
                                catch (Exception ex)
                                {
                                }
                            }
                        }
                    }
                    else
                    {
                        //7. diğer tüm pikseller siyah
                        Color c2 = Color.FromArgb(0, 0, 0);
                        mask.SetPixel(j, i, c2);
                        try
                        {

                        }
                        catch (Exception ex)
                        {
                        }
                    }
                }
            }
            return mask;
        }
        void Resize()
        {
            Bitmap bmp = new Bitmap(pictureBox1.Image, new Size(pictureBox1.Width, pictureBox1.Height)); pictureBox1.Image = bmp;
        }
        Bitmap Gray(Bitmap srcBmp)
        {
            Bitmap bmp = srcBmp;
            int NumRow = bmp.Height;
            int numCol = bmp.Width;
            Bitmap GRAY = new Bitmap(bmp.Width, bmp.Height);// GRAY matris sonucu 

            for (int i = 0; i < NumRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {
                    Color c = bmp.GetPixel(j, i);// renk pixsellerini çıkarma
                    int rd = c.R; int gr = c.G; int bl = c.B;// mavi kırmızı yeşil renkleri var
                    double d1 = 0.2989 * (double)rd + 0.5870 * (double)gr + 0.1140 * (double)bl;
                    int c1 = (int)Math.Round(d1);
                    Color c2 = Color.FromArgb(c1, c1, c1);
                    GRAY.SetPixel(j, i, c2);
                }
            }
            return GRAY;

        }
        double Bin2Dec(List<int> bin)
        {
            double d = 0;

            for (int i = 0; i < bin.Count; i++)
            {
                d += bin[i] * Math.Pow(2, i);
            }
            return d;
        }
        Bitmap LBP(Bitmap srcBmp, int R)
        {
            // srcBmp ve pencere R'den LBP görüntüsü almak istiyoruz
            Bitmap bmp = srcBmp;
            //. srcImage öğesinden satırları ve sütunları çıkarın. Not Kaynak görüntü Gri tonlamalı Dönüştürülen Görüntüdür
            int NumRow = srcBmp.Height;
            int numCol = srcBmp.Width;
            Bitmap lbp = new Bitmap(numCol, NumRow);
            Bitmap GRAY = new Bitmap(pictureBox1.Width, pictureBox1.Height);// GRİ sonuç matrisidir
            double[,] MAT = new double[numCol, NumRow];
            double max = 0.0;
            //. Piksellerde Döngü
            for (int i = 0; i < NumRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {
                    //  Color c1=Color.FromArgb(0,0,0);
                    MAT[j, i] = 0;

                    //sınır koşulunu tanımlıyruz yoksa (0,0) oluyor ve uygun komşuları olmadığını bulur
                    if ((i > R) && (j > R) && (i < (NumRow - R)) && (j < (numCol - R)))
                    {
                        // ikili değerleri bir Listede saklamak istiyoruz
                        List<int> vals = new List<int>();
                        try
                        {
                            for (int i1 = i - R; i1 < (i + R); i1++)
                            {
                                for (int j1 = j - R; j1 < (j + R); j1++)
                                {
                                    int acPixel = srcBmp.GetPixel(j, i).R;
                                    int nbrPixel = srcBmp.GetPixel(j1, i1).R;
                                    // 3. Bu, LBP'nin ana Mantığıdır
                                    if (nbrPixel > acPixel)
                                    {
                                        vals.Add(1);

                                    }
                                    else
                                    {
                                        vals.Add(0);
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                        }
                        ///4. 1'ler ve 0'lardan oluşan bir listemiz olduğunda, listeyi ondalık sayıya dönüştürdük
                        // Ayrıca normalleştirme amacıyla Max değeri hesaplamasını yapalım
                        double d1 = Bin2Dec(vals);
                        MAT[j, i] = d1;
                        if (d1 > max)
                        {
                            max = d1;
                        }
                    }
                }
            }
            //5. LBP matrisi MAT'yi normalleştirin ve LBP görüntüsü lbp elde edin
            lbp = NormalizeLbpMatrix(MAT, lbp, max);
            return lbp;
        }
        Bitmap NormalizeLbpMatrix(double[,] Mat, Bitmap lbp, double max)
        {
            int NumRow = lbp.Height;
            int numCol = lbp.Width;
            for (int i = 0; i < NumRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {

                    // pikseli maksimum değere bölüp 255 ile çarpmanın Normalleştirme yapıyoruz
                    double d = Mat[j, i] / max;
                    int v = (int)(d * 255);
                    Color c = Color.FromArgb(v, v, v);
                    lbp.SetPixel(j, i, c);
                }
            }
            return lbp;
        }
        Bitmap HOG(Bitmap img, int cellWidth, int cellHeight, int blockWidth, int blockHeight, bool signedOrientations = false)
        {

            Bitmap bmp = img;
            double[] ret = null;
            int imgWidth = bmp.Width;
            int imgHeight = bmp.Height;
            int[,] magnitudes = new int[imgHeight, imgWidth];
            int[,] directions = new int[imgHeight, imgWidth];

            Console.WriteLine("Directions: ");
            for (int i = 0; i < imgHeight; i++)
            {
                for (int j = 0; j < imgWidth; j++)
                {
                    //Sobel kısmının aynısı ilk aşama olarak
                    var pixelLeft = j - 1 < 0 ? Color.Black : bmp.GetPixel(j - 1, i);
                    var pixelUp = i + 1 >= imgHeight ? Color.Black : bmp.GetPixel(j, i + 1);
                    var pixelRight = j + 1 >= imgWidth ? Color.Black : bmp.GetPixel(j + 1, i);
                    var pixelDown = i - 1 < 0 ? Color.Black : bmp.GetPixel(j, i - 1);

                    int maxGradX = GetMaximalGradient(pixelLeft, pixelRight);
                    int maxGradY = GetMaximalGradient(pixelUp, pixelDown);

                    magnitudes[i, j] = (int)Math.Sqrt(maxGradX * maxGradX + maxGradY * maxGradY);
                    directions[i, j] = (int)((Math.Atan2(maxGradY, maxGradX) * 180.0 / Math.PI) + 180) % 180;
                    Console.Write(directions[i, j] + " ");
                }
                Console.WriteLine();
            }
            if (!SignedOrientations)
            {
                double[,] histograms = new double[(imgWidth / CellWidth) * (imgHeight / CellHeight), 9];

                for (int i = 0; i < imgHeight; i++)
                {
                    for (int j = 0; j < imgWidth; j++)
                    {
                        int cell = j / CellWidth + i / CellHeight * (imgWidth / CellWidth);
                        if (cell >= histograms.GetLength(0)) continue;
                        int directionFirstIndex = (directions[i, j] / 20) % 9;
                        int directionSecondIndex = (directionFirstIndex + 1) % 9;
                        double directionFirstValue = (1 - (directions[i, j] % 20) / 20) * magnitudes[i, j];
                        double directionSecondValue = magnitudes[i, j] - directionFirstValue;

                        histograms[cell, directionFirstIndex] += directionFirstValue;
                        histograms[cell, directionSecondIndex] += directionSecondValue;
                    }
                }
                Console.WriteLine("Histograms:");

                for (int i = 0; i < histograms.GetLength(0); i++)
                {
                    for (int j = 0; j < histograms.GetLength(1); j++)
                    {
                        Console.Write(histograms[i, j] + " ");
                    }
                    Console.WriteLine();
                }
                double[] finalVector = new double[((imgWidth / CellWidth) - BlockWidth + 1) * ((imgHeight / CellHeight) - BlockHeight + 1) * BlockWidth * BlockHeight * 9];
                int finalVectorCounter = 0;

                for (int i = 0; i + BlockWidth < (imgWidth / CellWidth); i++)
                {
                    for (int j = 0; j + BlockHeight < (imgHeight / CellHeight); j++)
                    {
                        double[] vector = new double[BlockWidth * BlockHeight * 9];
                        double vectorLength = 0;

                        for (int k = 0; k < vector.Length; k++)
                        {
                            int myCellNum = k / 9;
                            int cellNum = (j + myCellNum / BlockWidth) * (imgWidth / CellWidth) + i + myCellNum % BlockWidth;
                            vector[k] = histograms[cellNum, k % 9];
                            vectorLength += vector[k] * vector[k];
                        }
                        vectorLength = Math.Sqrt(vectorLength);

                        for (int k = 0; k < vector.Length; k++)
                        {
                            vector[k] /= vectorLength;
                            finalVector[finalVectorCounter] = vector[k];
                            finalVectorCounter++;
                        }
                    }

                }
                ret = finalVector;
            }
            return ret; //yeni atamış olduğumuz diziyi döndürmüyor
        }
        private int GetMaximalGradient(Color firstPixel, Color secondPixel)
        {
            int max = secondPixel.R - firstPixel.R;

            if (Math.Abs(secondPixel.G - firstPixel.G) > Math.Abs(max)) max = secondPixel.G - firstPixel.G;
            if (Math.Abs(secondPixel.B - firstPixel.B) > Math.Abs(max)) max = secondPixel.B - firstPixel.B;
            return max;
        }
        private void griButton_Click(object sender, EventArgs e)
        {
            if (!String.IsNullOrEmpty(selectedPath))
            {
                string[] files = Directory.GetFiles(selectedPath);

                if (files.Length > 0)
                {
                    if (!Directory.Exists("grayImages"))
                    {
                        Directory.CreateDirectory("grayImages");
                    }
                    foreach (var item in files)
                    {
                        try
                        {
                            pictureBox1.Image = Bitmap.FromFile(item);
                            pictureBox2.Image = Gray((Bitmap)pictureBox1.Image);
                            //Image = gry yapması için butona bastık bastıktan sonra gri oldu ve aynı pictureBox'da görüntüyü koydu
                            var l = item.Split('\\');
                            var lastName = l[l.Length - 1];

                            pictureBox2.Image.Save("grayImages\\" + lastName, ImageFormat.Jpeg);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show(item + " Gray İşleminde Hata!", "Message");
                        }
                    }
                }
            }
            else
            {
                MessageBox.Show("Lütfen Veri Seti Yolu Seçiniz!", "Message");
            }
        }

        //form içine ilki seçildiğnde diğer seçilmesi gereken koşullardan dolayı if bloklarını eklemeiz gerekir.

        private void classifierComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {

            if (classifierComboBox.SelectedIndex == 0)
            {
                pictureBox1.Image = LBP((Bitmap)pictureBox2.Image, int.Parse(textBox1.Text));

            }
            if (classifierComboBox.SelectedIndex == 1)
            {
                var image = new Bitmap(@"C:\Users\elif\Desktop\grayImages\\");
                var sw = new Stopwatch();
                sw.Start();
                pictureBox3.Image = HOG(image, 8, 8, 2, 2);
            }
            //seçilen algoritma ne ise ona göre uygulama yapılacak

        }
        private void SizeTextBox_TextChanged(object sender, EventArgs e)
        {
            //kullanıcıdan alına test_split boyut oranı alındı
            float gelen = 0;
            float.TryParse(SizeTextBox.Text, out gelen);
            //kullanıcıdan sayı al ve svm ,knn gönder 
        }
        //Resim için istenilen öznitelik çıkarma kısmı seçildi


        // Tüm istenilen durumlar seçildikten sonra uygula butonu ile istenilenler resme uygulandı
        private void applyButton_Click(object sender, EventArgs e)
        {

            if (!String.IsNullOrEmpty(selectedPath))
            {
                string[] files = Directory.GetFiles(selectedPath);

                if (files.Length > 0)
                {
                    if (!Directory.Exists("LBPImages"))
                    {
                        Directory.CreateDirectory("LBPImages");
                    }

                    foreach (var items in files)
                    {
                        try
                        {
                            pictureBox3.Image = LBP((Bitmap)pictureBox2.Image, int.Parse(textBox1.Text));
                            //Image = gry yapması için butona bastık bastıktan sonra gri oldu ve aynı pictureBox'da görüntüyü koydu
                            var l = items.Split('\\');
                            var lastName = l[l.Length - 1];

                            pictureBox3.Image.Save("LBPImages\\" + lastName, ImageFormat.Jpeg);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show(items + " LBP İşleminde Hata!", "Message");
                        }
                    }
                }
            }
            else
            {
                MessageBox.Show("Lütfen Veri Seti Yolu Seçiniz!", "Message");
            }

            ////Image = gry yapması için butona bastık bastıktan sonra gri oldu ve aynı pictureBox'da görüntüyü koydu
            DialogResult result;
            result = MessageBox.Show("Local Binary Pattern İşlemi Başarıyla Gerçekleşti", "Bilgi Paneli", MessageBoxButtons.OK, MessageBoxIcon.Information);
            if (result == DialogResult.OK)
            {
                //    //Uygulamadan çıkmaması için exit çalıştırmıyoruz
                // Application.Exit();
            }
        }

        //uygulama işlemi yapılan rrsimleri daha sonra test etmek için kaydetme işlemi gerçekleşti
        private void saveButton_Click(object sender, EventArgs e)
        {
            if (!String.IsNullOrEmpty(selectedPath))
            {
                string[] files = Directory.GetFiles(selectedPath);

                if (files.Length > 0)
                {
                    var s = new StringBuilder();
                    var i = 1;
                    foreach (var item in files)
                    {
                        s.AppendLine(i + " - " + item.ToString());
                        i++;
                    }
                    File.WriteAllText("temp.txt", s.ToString());
                    Console.WriteLine("çalıtşı");
                }
            }
            else
            {
                MessageBox.Show("Lütfen veri seti yolu seçiniz!", "Message");
            }
        }

        //kaydedilen resimlerin restgele bir tanesi seçilerek test aşamasına geçildi
        private void testImageUploadButton_Click(object sender, EventArgs e)
        {

        }
        //Test aşamasında olan resimlerin benzerlikleri bulundu ve başarı oranı hesaplandı
        private void hesaplaButton_Click(object sender, EventArgs e)
        {

        }
        //Başarı oranı hesaplandıktan sonra bu kısımda kullanıcya başarı oranı gösterildi
        private void accuracyLabel_Click(object sender, EventArgs e)
        {
        }
        private void featureLabel_Click(object sender, EventArgs e)
        {
        }
        //Aeçilen resim için bir sınıflandırıcı seçildi
        private void classifierLabel_Click(object sender, EventArgs e)
        {
            int threshold = 0;
            //  int.TryParse(ThresholdText.Text, out threshold);
        }
        //sınıflandırıc seçildikten sonra resimlerin kaçta kaçını kullanacaağımız belirtildi
        private void testSplitLabel_Click(object sender, EventArgs e)
        {

        }
        private void pictureBox2_Click(object sender, EventArgs e)
        {

        }
        private void pictureBox2_Click_1(object sender, EventArgs e)
        {

        }

        //Bu kısım confusion matrisi için geçerlidir


    }
}

    
