using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Teslim.Models;
using HOGDescriptor = Teslim.Models.HOGDescriptor;

namespace Teslim
{
    public partial class Form1 : Form
    {
        List<Data> DataSet = new List<Data>();
        List<Data> TrainingData;
        List<Data> TestingData;
        Matrix<float> x_train = null, x_test = null;
        Matrix<int> y_train = null, y_test = null;
        SVM svmModel;
        KNearest kNearestModel;

        List<int> PredictedLabels = null;
        List<int> ActualLabels = null;
        public Form1()
        {
            InitializeComponent();
        }

        private void messageLabel_Click(object sender, EventArgs e)
        {

        }

        private void dataLoadToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                FolderBrowserDialog dialog = new FolderBrowserDialog();
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    Cursor = Cursors.WaitCursor;

                    var files = Directory.GetFiles(dialog.SelectedPath);
                    foreach (var file in files)
                    {
                        var img = new Image<Gray, byte>(file).Resize(256, 256, Inter.Cubic);
                        var name = Path.GetFileName(file);
                        int label = int.Parse(name.Substring(name.IndexOf("_") - 2, 2));
                        var index = DataSet.FindIndex(x => x.Label == label);
                        if (index > -1)
                        {
                            DataSet[index].Images.Add(img);
                        }
                        else
                        {
                            Data data = new Data();
                            data.Images = new List<Image<Gray, byte>>();
                            data.Images.Add(img);
                            data.Label = label;
                            DataSet.Add(data);
                        }
                    }
                    messageLabel.Text = "Veri Seti Yüklendi.";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                Cursor = Cursors.Default;
            }
        }

        private void trainTestSplitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (DataSet == null || DataSet.Count == 0)
                {
                    throw new Exception("Veri Seti Boş");
                }

                (TrainingData, TestingData) = Connet.TestTrainSplit(DataSet);
                messageLabel.Text = "Test-Train Split Başarılı.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void sVMTrainToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                Cursor = Cursors.WaitCursor;
                messageLabel.Text = "";
                messageLabel.ForeColor = Color.Black;


                if (x_train == null || x_train.Rows < 1)
                {
                    throw new Exception("Öznitelikler Train Aşaması için Eklendi.");
                }

                svmModel = new SVM();
                kNearestModel = new KNearest();
                if (File.Exists("data_rbf_svm"))
                {
                    svmModel.Load("data_svm");
                    messageLabel.Text = " SVM Eğitim Modeli Yüklendi.";
                }
                else
                {
                    svmModel.SetKernel(SVM.SvmKernelType.Rbf);
                    svmModel.Type = SVM.SvmType.CSvc;
                    svmModel.TermCriteria = new MCvTermCriteria(1000, 0.00001);
                    svmModel.C = 250;
                    svmModel.Gamma = 0.001;

                    TrainData traindata = new TrainData(x_train, DataLayoutType.RowSample, y_train);
                    if (svmModel.Train(traindata))
                    {
                        svmModel.Save("data_rbf_svm");
                        messageLabel.Text = "SVM Model Trained & Kaydedildi.";
                    }
                    else
                    {
                        messageLabel.Text = "SVM Model eğitimi Başarısız.";
                        messageLabel.ForeColor = Color.Red;
                    }

                }



            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                Cursor = Cursors.Default;

            }
        }

        private void sVMTestToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (svmModel == null)
                {
                    throw new Exception("SVM Eğitim Yapılacak Model Yok.");
                }

                (x_test, y_test) = CalculateHoGFeatures(TestingData);

                if (x_test == null || x_test.Rows < 1)
                {
                    throw new Exception("Test Datası Yüklendi.");
                }


                PredictedLabels = new List<int>();
                ActualLabels = new List<int>();


                for (int i = 0; i < x_test.Rows; i++)
                {
                    var prediction = svmModel.Predict(x_test.GetRow(i));
                    PredictedLabels.Add((int)prediction);
                    ActualLabels.Add(y_test[i, 0]);
                }

                var cm = Connet.ComputeConfusionMatrix(ActualLabels.ToArray(), PredictedLabels.ToArray());
                var metrics = Connet.CalculateMetrics(cm, ActualLabels.ToArray(), PredictedLabels.ToArray());
                string results = $"Test Örnek Sayısı = {ActualLabels.Count} \n Accuracy = {metrics[0] * 100}% " +
                    $"\nPrecision = {metrics[1] * 100}% \n Recall = {metrics[2] * 100}%";

                FormConfusionMatrix form = new FormConfusionMatrix(cm, results);
                form.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void showResultToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            try
            {
                if (PredictedLabels == null || ActualLabels == null)
                {
                    throw new Exception("aranılan Tahmin Bulunamadı.");
                }

                Random random = new Random();
                int index = random.Next(PredictedLabels.Count - 1);
                int predLabel = PredictedLabels[index];
                int actualLabel = ActualLabels[index];

                var predImage = (from img in TestingData
                                 where img.Label == predLabel
                                 select img.Images[random.Next(img.Images.Count)])
                                .FirstOrDefault().Clone();

                var actualImage = (from img in TestingData
                                   where img.Label == actualLabel
                                   select img.Images[random.Next(img.Images.Count)])
                                .FirstOrDefault().Clone();
                CvInvoke.PutText(predImage, "Predicted", new Point(30, 30), FontFace.HersheyPlain, 1.0, new MCvScalar(0));
                CvInvoke.PutText(actualImage, "Actual", new Point(30, 30), FontFace.HersheyPlain, 1.0, new MCvScalar(0));

                var imgOutput = Connet.HConcatenateImages(predImage.Convert<Bgr, byte>(), actualImage.Convert<Bgr, byte>());
                pictureBox1.Image = imgOutput.AsBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void kNNTrainToolStripMenuItem_Click(object sender, EventArgs e)//linear svm göre işlem yapıyoruz
        {
            try
            {
                Cursor = Cursors.WaitCursor;
                messageLabel.Text = "";
                messageLabel.ForeColor = Color.Black;


                if (x_train == null || x_train.Rows < 1)
                {
                    throw new Exception("Linear Kernel için Öznitelikler Yüklendi.");
                }

                svmModel = new SVM();
                if (File.Exists("data_linear_svm"))
                {
                    svmModel.Load("data_linear_svm");
                    messageLabel.Text = " LİNEAR SVM Eğiitim Modeli Yüklendi.";
                }
                else
                {
                    svmModel.SetKernel(SVM.SvmKernelType.Linear);
                    svmModel.Type = SVM.SvmType.CSvc;
                    svmModel.TermCriteria = new MCvTermCriteria(1000, 0.00001);
                    svmModel.C = 250;
                    svmModel.Gamma = 0.001;

                    TrainData traindata = new TrainData(x_train, DataLayoutType.RowSample, y_train);
                    if (svmModel.Train(traindata))
                    {
                        svmModel.Save("data_linear_svm");
                        messageLabel.Text = "LİNEAR SVM  Eğitim Modeli & Kaydedildi.";
                    }
                    else
                    {
                        messageLabel.Text = "SVM Linear Model Eğitimi Başarısız.";
                        messageLabel.ForeColor = Color.Red;
                    }

                }



            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                Cursor = Cursors.Default;

            }
        }

        private void kNNTestToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (svmModel == null)
                {
                    throw new Exception(" LİNEAR SVM Eğitilmedi.");
                }

                (x_test, y_test) = CalculateHoGFeatures(TestingData);

                if (x_test == null || x_test.Rows < 1)
                {
                    throw new Exception("Test Datası Yüklendi.");
                }


                PredictedLabels = new List<int>();
                ActualLabels = new List<int>();


                for (int i = 0; i < x_test.Rows; i++)
                {
                    var prediction = svmModel.Predict(x_test.GetRow(i));
                    PredictedLabels.Add((int)prediction);
                    ActualLabels.Add(y_test[i, 0]);
                }

                var cm = Connet.ComputeConfusionMatrix(ActualLabels.ToArray(), PredictedLabels.ToArray());
                var metrics = Connet.CalculateMetrics(cm, ActualLabels.ToArray(), PredictedLabels.ToArray());
                string results = $"Test Örnek Sayısı = {ActualLabels.Count} \n Accuracy = {metrics[0] * 100}% " +
                    $"\nPrecision = {metrics[1] * 100}% \n Recall = {metrics[2] * 100}%";

                FormConfusionMatrix form = new FormConfusionMatrix(cm, results);
                form.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        private void hOGToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (TrainingData == null || TestingData == null)
                {
                    throw new Exception("Test or Train data not found.");
                }
                Cursor = Cursors.WaitCursor;
                (x_train, y_train) = CalculateHoGFeatures(TrainingData);
                messageLabel.Text = "Training: Hog extracted.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                Cursor = Cursors.Default;
            }
        }

        private static (Matrix<float>, Matrix<int>) CalculateHoGFeatures(List<Data> data)
        {
            try
            {
                HOGDescriptor hogs = new HOGDescriptor();
                double[] d = hogs.Describe();

                List<float[]> finalVector = new List<float[]>();
                List<int> labels = new List<int>();

                foreach (var item in data)
                {
                    foreach (var img in item.Images)
                    {
                        var features = hogs.Compute(img);
                        finalVector.Add((float[])features);
                        labels.Add(item.Label);
                    }

                    var xtrain = new Matrix<float>(Connet.To2D<float>(finalVector.ToArray()));
                    var ytrain = new Matrix<int>(labels.ToArray());
                    return (xtrain, ytrain);


                }
            }
            catch (Exception)
            {

                throw;
            }
        }


        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
