using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace ImageProcessingProject
{
    public partial class SplashForm : Form
    {
        public SplashForm()
        {
            InitializeComponent();
        }
        private void timer1_Tick(object sender, EventArgs e)
      {
           progressBar1.Increment(5);
           if (progressBar1.Value == 100)
            {
            timer1.Stop();
               this.Close();
              AnaEkran frmae1 = new AnaEkran();
              frmae1.Show();
           }
       }

      private void SplashForm_Load(object sender, EventArgs e)      {
     }
 }
}




