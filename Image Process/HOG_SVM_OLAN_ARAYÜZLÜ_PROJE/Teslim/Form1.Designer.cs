
namespace Teslim
{
    partial class Form1
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
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.messageLabel = new System.Windows.Forms.Label();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.dataLoadToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.trainTestSplitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.featuresToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.hOGToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.sVMTrainToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.sVMTestToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.kNNTrainToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.kNNTestToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.lBPToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.showResultToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(203, 83);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(438, 191);
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // messageLabel
            // 
            this.messageLabel.AutoSize = true;
            this.messageLabel.Location = new System.Drawing.Point(364, 317);
            this.messageLabel.Name = "messageLabel";
            this.messageLabel.Size = new System.Drawing.Size(0, 15);
            this.messageLabel.TabIndex = 1;
            this.messageLabel.Click += new System.EventHandler(this.messageLabel_Click);
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.dataLoadToolStripMenuItem,
            this.showResultToolStripMenuItem1});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(800, 24);
            this.menuStrip1.TabIndex = 2;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // dataLoadToolStripMenuItem
            // 
            this.dataLoadToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.trainTestSplitToolStripMenuItem});
            this.dataLoadToolStripMenuItem.Name = "dataLoadToolStripMenuItem";
            this.dataLoadToolStripMenuItem.Size = new System.Drawing.Size(72, 20);
            this.dataLoadToolStripMenuItem.Text = "Data Load";
            this.dataLoadToolStripMenuItem.Click += new System.EventHandler(this.dataLoadToolStripMenuItem_Click);
            // 
            // trainTestSplitToolStripMenuItem
            // 
            this.trainTestSplitToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.featuresToolStripMenuItem,
            this.hOGToolStripMenuItem,
            this.lBPToolStripMenuItem});
            this.trainTestSplitToolStripMenuItem.Name = "trainTestSplitToolStripMenuItem";
            this.trainTestSplitToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.trainTestSplitToolStripMenuItem.Text = "Train Test Split";
            this.trainTestSplitToolStripMenuItem.Click += new System.EventHandler(this.trainTestSplitToolStripMenuItem_Click);
            // 
            // featuresToolStripMenuItem
            // 
            this.featuresToolStripMenuItem.Name = "featuresToolStripMenuItem";
            this.featuresToolStripMenuItem.Size = new System.Drawing.Size(118, 22);
            this.featuresToolStripMenuItem.Text = "Features";
            // 
            // hOGToolStripMenuItem
            // 
            this.hOGToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.sVMTrainToolStripMenuItem,
            this.kNNTrainToolStripMenuItem});
            this.hOGToolStripMenuItem.Name = "hOGToolStripMenuItem";
            this.hOGToolStripMenuItem.Size = new System.Drawing.Size(118, 22);
            this.hOGToolStripMenuItem.Text = "HOG";
            this.hOGToolStripMenuItem.Click += new System.EventHandler(this.hOGToolStripMenuItem_Click);
            // 
            // sVMTrainToolStripMenuItem
            // 
            this.sVMTrainToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.sVMTestToolStripMenuItem});
            this.sVMTrainToolStripMenuItem.Name = "sVMTrainToolStripMenuItem";
            this.sVMTrainToolStripMenuItem.Size = new System.Drawing.Size(127, 22);
            this.sVMTrainToolStripMenuItem.Text = "SVM Train";
            this.sVMTrainToolStripMenuItem.Click += new System.EventHandler(this.sVMTrainToolStripMenuItem_Click);
            // 
            // sVMTestToolStripMenuItem
            // 
            this.sVMTestToolStripMenuItem.Name = "sVMTestToolStripMenuItem";
            this.sVMTestToolStripMenuItem.Size = new System.Drawing.Size(121, 22);
            this.sVMTestToolStripMenuItem.Text = "SVM Test";
            this.sVMTestToolStripMenuItem.Click += new System.EventHandler(this.sVMTestToolStripMenuItem_Click);
            // 
            // kNNTrainToolStripMenuItem
            // 
            this.kNNTrainToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.kNNTestToolStripMenuItem});
            this.kNNTrainToolStripMenuItem.Name = "kNNTrainToolStripMenuItem";
            this.kNNTrainToolStripMenuItem.Size = new System.Drawing.Size(127, 22);
            this.kNNTrainToolStripMenuItem.Text = "KNN Train";
            this.kNNTrainToolStripMenuItem.Click += new System.EventHandler(this.kNNTrainToolStripMenuItem_Click);
            // 
            // kNNTestToolStripMenuItem
            // 
            this.kNNTestToolStripMenuItem.Name = "kNNTestToolStripMenuItem";
            this.kNNTestToolStripMenuItem.Size = new System.Drawing.Size(122, 22);
            this.kNNTestToolStripMenuItem.Text = "KNN Test";
            this.kNNTestToolStripMenuItem.Click += new System.EventHandler(this.kNNTestToolStripMenuItem_Click);
            // 
            // lBPToolStripMenuItem
            // 
            this.lBPToolStripMenuItem.Name = "lBPToolStripMenuItem";
            this.lBPToolStripMenuItem.Size = new System.Drawing.Size(118, 22);
            this.lBPToolStripMenuItem.Text = "LBP";
            // 
            // showResultToolStripMenuItem1
            // 
            this.showResultToolStripMenuItem1.Name = "showResultToolStripMenuItem1";
            this.showResultToolStripMenuItem1.Size = new System.Drawing.Size(83, 20);
            this.showResultToolStripMenuItem1.Text = "Show Result";
            this.showResultToolStripMenuItem1.Click += new System.EventHandler(this.showResultToolStripMenuItem1_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.messageLabel);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
            this.Text = "Görüntü İşleme Projesi";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Label messageLabel;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem dataLoadToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem trainTestSplitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem featuresToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem hOGToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem sVMTrainToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem sVMTestToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem lBPToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem showResultToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem kNNTrainToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem kNNTestToolStripMenuItem;
    }
}

