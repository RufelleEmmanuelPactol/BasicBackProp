namespace BackPropagationGates
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
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
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.hLayerCounter = new System.Windows.Forms.Label();
            this.epochCounter = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.metric = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.button2 = new System.Windows.Forms.Button();
            this.label12 = new System.Windows.Forms.Label();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.button3 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.button6 = new System.Windows.Forms.Button();
            this.button7 = new System.Windows.Forms.Button();
            this.label15 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(17, 23);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(203, 52);
            this.button1.TabIndex = 0;
            this.button1.Text = "Optimize Accordiningly";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(17, 138);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(183, 24);
            this.label1.TabIndex = 1;
            this.label1.Text = "Current Hidden Layers:";
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(17, 162);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(183, 24);
            this.label2.TabIndex = 2;
            this.label2.Text = "Current Epoch Count: ";
            // 
            // hLayerCounter
            // 
            this.hLayerCounter.Location = new System.Drawing.Point(191, 138);
            this.hLayerCounter.Name = "hLayerCounter";
            this.hLayerCounter.Size = new System.Drawing.Size(183, 24);
            this.hLayerCounter.TabIndex = 3;
            this.hLayerCounter.Text = "0";
            // 
            // epochCounter
            // 
            this.epochCounter.Location = new System.Drawing.Point(191, 162);
            this.epochCounter.Name = "epochCounter";
            this.epochCounter.Size = new System.Drawing.Size(183, 24);
            this.epochCounter.TabIndex = 4;
            this.epochCounter.Text = "0";
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(17, 186);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(183, 24);
            this.label3.TabIndex = 5;
            this.label3.Text = "Accuracy: ";
            // 
            // metric
            // 
            this.metric.Location = new System.Drawing.Point(191, 186);
            this.metric.Name = "metric";
            this.metric.Size = new System.Drawing.Size(183, 24);
            this.metric.TabIndex = 6;
            this.metric.Text = "0";
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(191, 308);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(183, 24);
            this.label4.TabIndex = 12;
            this.label4.Text = "0";
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(17, 308);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(183, 24);
            this.label5.TabIndex = 11;
            this.label5.Text = "Accuracy: ";
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(191, 284);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(183, 24);
            this.label6.TabIndex = 10;
            this.label6.Text = "0";
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(191, 260);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(183, 24);
            this.label7.TabIndex = 9;
            this.label7.Text = "0";
            // 
            // label8
            // 
            this.label8.Location = new System.Drawing.Point(17, 284);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(183, 24);
            this.label8.TabIndex = 8;
            this.label8.Text = "Current Epoch Count: ";
            // 
            // label9
            // 
            this.label9.Location = new System.Drawing.Point(17, 260);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(183, 24);
            this.label9.TabIndex = 7;
            this.label9.Text = "Current Hidden Layers:";
            // 
            // label10
            // 
            this.label10.Location = new System.Drawing.Point(17, 236);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(183, 24);
            this.label10.TabIndex = 13;
            this.label10.Text = "Thread 2";
            // 
            // label11
            // 
            this.label11.Location = new System.Drawing.Point(17, 114);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(183, 24);
            this.label11.TabIndex = 14;
            this.label11.Text = "Thread 1";
            // 
            // button2
            // 
            this.button2.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button2.Location = new System.Drawing.Point(467, 89);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(66, 61);
            this.button2.TabIndex = 15;
            this.button2.Text = "0";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // label12
            // 
            this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label12.Location = new System.Drawing.Point(467, 62);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(183, 24);
            this.label12.TabIndex = 19;
            this.label12.Text = "INPUTS";
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(596, 138);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(90, 26);
            this.textBox1.TabIndex = 20;
            // 
            // label13
            // 
            this.label13.Location = new System.Drawing.Point(596, 111);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(183, 24);
            this.label13.TabIndex = 21;
            this.label13.Text = "Assumed Value:";
            // 
            // label14
            // 
            this.label14.Location = new System.Drawing.Point(596, 186);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(183, 24);
            this.label14.TabIndex = 22;
            this.label14.Text = "Discretized Result:";
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(596, 213);
            this.textBox2.Name = "textBox2";
            this.textBox2.ReadOnly = true;
            this.textBox2.Size = new System.Drawing.Size(90, 26);
            this.textBox2.TabIndex = 23;
            // 
            // button3
            // 
            this.button3.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button3.Location = new System.Drawing.Point(467, 156);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(66, 61);
            this.button3.TabIndex = 24;
            this.button3.Text = "0";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // button4
            // 
            this.button4.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button4.Location = new System.Drawing.Point(467, 223);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(66, 61);
            this.button4.TabIndex = 25;
            this.button4.Text = "0";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button5
            // 
            this.button5.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button5.Location = new System.Drawing.Point(467, 290);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(66, 61);
            this.button5.TabIndex = 26;
            this.button5.Text = "0";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(596, 274);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(101, 34);
            this.button6.TabIndex = 27;
            this.button6.Text = "Train First";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(596, 314);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(101, 34);
            this.button7.TabIndex = 28;
            this.button7.Text = "Train First";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            // 
            // label15
            // 
            this.label15.Location = new System.Drawing.Point(596, 248);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(183, 24);
            this.label15.TabIndex = 29;
            this.label15.Text = "Run Network";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.Control;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.button7);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.metric);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.epochCounter);
            this.Controls.Add(this.hLayerCounter);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.button1);
            this.Location = new System.Drawing.Point(15, 15);
            this.Name = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.Label label15;

        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;

        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox textBox2;

        private System.Windows.Forms.Label label12;

        private System.Windows.Forms.Button button2;

        private System.Windows.Forms.Label label11;

        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label10;

        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label metric;

        private System.Windows.Forms.Label hLayerCounter;
        private System.Windows.Forms.Label epochCounter;

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;

        private System.Windows.Forms.Button button1;

        #endregion
    }
}