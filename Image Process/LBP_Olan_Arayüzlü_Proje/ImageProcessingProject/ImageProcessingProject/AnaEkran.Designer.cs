
namespace ImageProcessingProject
{
    partial class AnaEkran
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AnaEkran));
            this.ımageList1 = new System.Windows.Forms.ImageList(this.components);
            this.accuracyLabel = new System.Windows.Forms.Label();
            this.testSplitLabel = new System.Windows.Forms.Label();
            this.classifierLabel = new System.Windows.Forms.Label();
            this.featureLabel = new System.Windows.Forms.Label();
            this.aploadButton = new System.Windows.Forms.Button();
            this.applyButton = new System.Windows.Forms.Button();
            this.saveButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.griButton = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.testImageUploadButton = new System.Windows.Forms.Button();
            this.hesaplaButton = new System.Windows.Forms.Button();
            this.classifierComboBox = new System.Windows.Forms.ComboBox();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.SizeTextBox = new System.Windows.Forms.TextBox();
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            // 
            // ımageList1
            // 
            this.ımageList1.ColorDepth = System.Windows.Forms.ColorDepth.Depth8Bit;
            this.ımageList1.ImageSize = new System.Drawing.Size(16, 16);
            this.ımageList1.TransparentColor = System.Drawing.Color.Transparent;
            // 
            // accuracyLabel
            // 
            this.accuracyLabel.AutoSize = true;
            this.accuracyLabel.BackColor = System.Drawing.Color.Transparent;
            this.accuracyLabel.Font = new System.Drawing.Font("Segoe UI Black", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.accuracyLabel.ForeColor = System.Drawing.Color.SkyBlue;
            this.accuracyLabel.Location = new System.Drawing.Point(708, 461);
            this.accuracyLabel.Name = "accuracyLabel";
            this.accuracyLabel.Size = new System.Drawing.Size(106, 21);
            this.accuracyLabel.TabIndex = 8;
            this.accuracyLabel.Text = "Başarı Oranı";
            this.accuracyLabel.Click += new System.EventHandler(this.accuracyLabel_Click);
            // 
            // testSplitLabel
            // 
            this.testSplitLabel.AutoSize = true;
            this.testSplitLabel.BackColor = System.Drawing.Color.Transparent;
            this.testSplitLabel.Font = new System.Drawing.Font("Segoe UI Black", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.testSplitLabel.ForeColor = System.Drawing.Color.SkyBlue;
            this.testSplitLabel.Location = new System.Drawing.Point(12, 351);
            this.testSplitLabel.Name = "testSplitLabel";
            this.testSplitLabel.Size = new System.Drawing.Size(137, 21);
            this.testSplitLabel.TabIndex = 7;
            this.testSplitLabel.Text = "Test Boyutu (%)";
            this.testSplitLabel.Click += new System.EventHandler(this.testSplitLabel_Click);
            // 
            // classifierLabel
            // 
            this.classifierLabel.AutoSize = true;
            this.classifierLabel.BackColor = System.Drawing.Color.Transparent;
            this.classifierLabel.Font = new System.Drawing.Font("Segoe UI Black", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.classifierLabel.ForeColor = System.Drawing.Color.SkyBlue;
            this.classifierLabel.Location = new System.Drawing.Point(12, 431);
            this.classifierLabel.Name = "classifierLabel";
            this.classifierLabel.Size = new System.Drawing.Size(110, 21);
            this.classifierLabel.TabIndex = 6;
            this.classifierLabel.Text = "Sınıflandırıcı";
            this.classifierLabel.Click += new System.EventHandler(this.classifierLabel_Click);
            // 
            // featureLabel
            // 
            this.featureLabel.AutoSize = true;
            this.featureLabel.BackColor = System.Drawing.Color.Transparent;
            this.featureLabel.Font = new System.Drawing.Font("Segoe UI Black", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.featureLabel.ForeColor = System.Drawing.Color.SkyBlue;
            this.featureLabel.Location = new System.Drawing.Point(12, 390);
            this.featureLabel.Name = "featureLabel";
            this.featureLabel.Size = new System.Drawing.Size(153, 21);
            this.featureLabel.TabIndex = 5;
            this.featureLabel.Text = "LBP Windows Size";
            this.featureLabel.Click += new System.EventHandler(this.featureLabel_Click);
            // 
            // aploadButton
            // 
            this.aploadButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.aploadButton.Location = new System.Drawing.Point(28, 26);
            this.aploadButton.Name = "aploadButton";
            this.aploadButton.Size = new System.Drawing.Size(75, 35);
            this.aploadButton.TabIndex = 11;
            this.aploadButton.Text = "Yükle";
            this.aploadButton.UseVisualStyleBackColor = false;
            this.aploadButton.Click += new System.EventHandler(this.uploadButton_Click);
            // 
            // applyButton
            // 
            this.applyButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.applyButton.Location = new System.Drawing.Point(644, 26);
            this.applyButton.Name = "applyButton";
            this.applyButton.Size = new System.Drawing.Size(75, 34);
            this.applyButton.TabIndex = 12;
            this.applyButton.Text = "Uygula";
            this.applyButton.UseVisualStyleBackColor = false;
            this.applyButton.Click += new System.EventHandler(this.applyButton_Click);
            // 
            // saveButton
            // 
            this.saveButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.saveButton.Location = new System.Drawing.Point(870, 12);
            this.saveButton.Name = "saveButton";
            this.saveButton.Size = new System.Drawing.Size(80, 34);
            this.saveButton.TabIndex = 13;
            this.saveButton.Text = "Kaydet";
            this.saveButton.UseVisualStyleBackColor = false;
            this.saveButton.Click += new System.EventHandler(this.saveButton_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.Font = new System.Drawing.Font("Segoe UI Black", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.label1.ForeColor = System.Drawing.SystemColors.ActiveCaption;
            this.label1.Location = new System.Drawing.Point(708, 408);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(134, 25);
            this.label1.TabIndex = 16;
            this.label1.Text = "TEST EKRANI";
            // 
            // griButton
            // 
            this.griButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.griButton.Location = new System.Drawing.Point(320, 26);
            this.griButton.Name = "griButton";
            this.griButton.Size = new System.Drawing.Size(132, 35);
            this.griButton.TabIndex = 20;
            this.griButton.Text = "Gry Ön işleme";
            this.griButton.UseVisualStyleBackColor = false;
            this.griButton.Click += new System.EventHandler(this.griButton_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.Color.Transparent;
            this.pictureBox1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox1.Location = new System.Drawing.Point(28, 67);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(244, 241);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 14;
            this.pictureBox1.TabStop = false;
            // 
            // testImageUploadButton
            // 
            this.testImageUploadButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.testImageUploadButton.Location = new System.Drawing.Point(870, 337);
            this.testImageUploadButton.Name = "testImageUploadButton";
            this.testImageUploadButton.Size = new System.Drawing.Size(75, 35);
            this.testImageUploadButton.TabIndex = 21;
            this.testImageUploadButton.Text = "Yükle";
            this.testImageUploadButton.UseVisualStyleBackColor = false;
            this.testImageUploadButton.Click += new System.EventHandler(this.testImageUploadButton_Click);
            // 
            // hesaplaButton
            // 
            this.hesaplaButton.BackColor = System.Drawing.Color.LightSkyBlue;
            this.hesaplaButton.Location = new System.Drawing.Point(754, 337);
            this.hesaplaButton.Name = "hesaplaButton";
            this.hesaplaButton.Size = new System.Drawing.Size(75, 35);
            this.hesaplaButton.TabIndex = 22;
            this.hesaplaButton.Text = "Hesapla";
            this.hesaplaButton.UseVisualStyleBackColor = false;
            this.hesaplaButton.Click += new System.EventHandler(this.hesaplaButton_Click);
            // 
            // classifierComboBox
            // 
            this.classifierComboBox.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.classifierComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.classifierComboBox.FormattingEnabled = true;
            this.classifierComboBox.Location = new System.Drawing.Point(177, 429);
            this.classifierComboBox.Name = "classifierComboBox";
            this.classifierComboBox.Size = new System.Drawing.Size(121, 23);
            this.classifierComboBox.TabIndex = 24;
            this.classifierComboBox.SelectedIndexChanged += new System.EventHandler(this.classifierComboBox_SelectedIndexChanged);
            this.classifierComboBox.FormattingEnabled = true;
            this.classifierComboBox.Items.AddRange(new object[] {
            "LBP",
            "HOG"});

            // 
            // textBox2
            // 
            this.textBox2.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.textBox2.Location = new System.Drawing.Point(824, 461);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(121, 23);
            this.textBox2.TabIndex = 26;
            // 
            // SizeTextBox
            // 
            this.SizeTextBox.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.SizeTextBox.Location = new System.Drawing.Point(177, 351);
            this.SizeTextBox.Name = "SizeTextBox";
            this.SizeTextBox.Size = new System.Drawing.Size(121, 23);
            this.SizeTextBox.TabIndex = 27;
            this.SizeTextBox.TextChanged += new System.EventHandler(this.SizeTextBox_TextChanged);
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(61, 4);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(177, 388);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(121, 23);
            this.textBox1.TabIndex = 28;
            // 
            // pictureBox3
            // 
            this.pictureBox3.BackColor = System.Drawing.Color.Transparent;
            this.pictureBox3.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox3.Location = new System.Drawing.Point(570, 67);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(244, 241);
            this.pictureBox3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox3.TabIndex = 29;
            this.pictureBox3.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.BackColor = System.Drawing.Color.Transparent;
            this.pictureBox2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox2.Location = new System.Drawing.Point(292, 67);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(244, 241);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox2.TabIndex = 30;
            this.pictureBox2.TabStop = false;
            this.pictureBox2.Click += new System.EventHandler(this.pictureBox2_Click_1);
            // 
            // AnaEkran
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("$this.BackgroundImage")));
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.ClientSize = new System.Drawing.Size(1165, 507);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.pictureBox3);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.SizeTextBox);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.classifierComboBox);
            this.Controls.Add(this.hesaplaButton);
            this.Controls.Add(this.testImageUploadButton);
            this.Controls.Add(this.griButton);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.saveButton);
            this.Controls.Add(this.applyButton);
            this.Controls.Add(this.aploadButton);
            this.Controls.Add(this.testSplitLabel);
            this.Controls.Add(this.accuracyLabel);
            this.Controls.Add(this.classifierLabel);
            this.Controls.Add(this.featureLabel);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "AnaEkran";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Image Processing Project";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Label accuracyLabel;
        private System.Windows.Forms.Label testSplitLabel;
        private System.Windows.Forms.Label classifierLabel;
        private System.Windows.Forms.Label featureLabel;
        private System.Windows.Forms.ImageList ımageList1;
        private System.Windows.Forms.Button aploadButton;
        private System.Windows.Forms.Button applyButton;
        private System.Windows.Forms.Button saveButton;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button griButton;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button testImageUploadButton;
        private System.Windows.Forms.Button hesaplaButton;
        private System.Windows.Forms.ComboBox classifierComboBox;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.TextBox SizeTextBox;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.PictureBox pictureBox3;
        private System.Windows.Forms.PictureBox pictureBox2;
    }
}

