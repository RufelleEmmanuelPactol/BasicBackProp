using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Backprop;

namespace BackPropagationGates
{
  public partial class Form1 : Form
  {

    private NeuralNet NeuralNetwork;
    private NeuralNet NeuralNetwork2;
    private int TotalEpochs;
    private double LowerThreshold = 0.05; // if we hit this, we are now at an optimal state
    private double AcceptanceThresAhHold = 0.95; // once we hit this, we are now at an optimal state
    private int MaxEpochs = 300000; // if we hit this and we have not stopped yet, we are not going to hit the threshold
    private int HiddenLayersSoFar;

    private static readonly List<Tuple<int, int, int, int>> TrainingData = new List<Tuple<int, int, int, int>>();


    private void InitializeTrainingData()
    {
      TrainingData.Add(new Tuple<int, int, int, int>(0, 0, 0, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 0, 0, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 0, 1, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 0, 1, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 1, 0, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 1, 0, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 1, 1, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(0, 1, 1, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 0, 0, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 0, 0, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 0, 1, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 0, 1, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 1, 0, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 1, 0, 0));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 1, 1, 1));
      TrainingData.Add(new Tuple<int, int, int, int>(1, 1, 1, 0));
    }

    private static readonly object lockObject = new object();

    // performs one epoch :)
    private void PerformEpoch(int threadCount)
    {
      NeuralNet nn = threadCount == 1 ? this.NeuralNetwork : this.NeuralNetwork2;
      for (int i = 0; i < 1; i++)
      {
        learnInputs(nn,0.0, 0.0, 0.0, 0.0);
        learnInputs(nn,0.0, 0.0, 0.0, 1.0);
        learnInputs(nn,0.0, 0.0, 1.0, 0.0);
        learnInputs(nn,0.0, 0.0, 1.0, 1.0);
        learnInputs(nn,0.0, 1.0, 0.0, 0.0);
        learnInputs(nn,0.0, 1.0, 0.0, 1.0);
        learnInputs(nn,0.0, 1.0, 1.0, 0.0);
        learnInputs(nn,0.0, 1.0, 1.0, 1.0);
        learnInputs(nn,1.0, 0.0, 0.0, 0.0);
        learnInputs(nn,1.0, 0.0, 0.0, 1.0);
        learnInputs(nn,1.0, 0.0, 1.0, 0.0);
        learnInputs(nn,1.0, 0.0, 1.0, 1.0);
        learnInputs(nn,1.0, 1.0, 0.0, 0.0);
        learnInputs(nn,1.0, 1.0, 0.0, 1.0);
        learnInputs(nn,1.0, 1.0, 1.0, 0.0);
        learnInputs(nn,1.0, 1.0, 1.0, 1.0);
  
      }
      nn.setInputs(0, 1);
      nn.setInputs(1, 1);
      nn.setInputs(2, 1);
      nn.setInputs(3, 1);
        nn.run();
      if (nn == NeuralNetwork)
      {
        metric.Text = Math.Round((nn.getOuputData(0) * 100), 2).ToString() + "% ";
      }
      else
      {
        label4.Text = Math.Round((nn.getOuputData(0) * 100), 2).ToString() + "%";
      }
      
      
      

      foreach (var tuple in TrainingData)
      {
        TrainNetwork(tuple.Item1, tuple.Item2, tuple.Item3, tuple.Item4, threadCount);
      }
    }
    
    private void learnInputs(NeuralNet nn, double in0, double in1, double in2, double in3)
    {
      nn.setInputs(0, in0);
      nn.setInputs(1, in1);
      nn.setInputs(2, in2);
      nn.setInputs(3, in3);
      if(in0 == 1.0 && in1 == 1.0 && in2 == 1.0 && in3 == 1.0)
        nn.setDesiredOutput(0, 1.0);
      else
        nn.setDesiredOutput(0, 0.0);
      nn.learn();
    }

    private int SystemEpochsSoFar = 0;


    private int evaluated = 0;

    private bool IsViable(NeuralNet neuralNet)
    {
      bool viable = false;
      neuralNet.setInputs(0, 0);
      neuralNet.setInputs(1, 0);
      neuralNet.setInputs(2, 0);
      neuralNet.setInputs(3, 0);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      
      var saved =  neuralNet.getOuputData(0);
      // Test the network with various input combinations
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();
     
      

      if (saved < LowerThreshold && neuralNet.getOuputData(0) > AcceptanceThresAhHold) return true;
      Refresh();
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 1);
      neuralNet.learn();
        

      
      neuralNet.setInputs(0, 0);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      
      var exp = neuralNet.getOuputData(0);
      if (exp > LowerThreshold)
      {
        return false;
      }
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 0);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      exp = neuralNet.getOuputData(0);
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      if (exp > LowerThreshold)
      {
        return false;
      }
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 0);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      exp = neuralNet.getOuputData(0);
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      if (exp > LowerThreshold)
      {
        return false;
      }
      
      neuralNet.setInputs(0, 1);
      neuralNet.setInputs(1, 1);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 0);
      neuralNet.run();
      exp = neuralNet.getOuputData(0);
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      if (exp > LowerThreshold)
      {
        return false;
      }
      neuralNet.setInputs(0, 0);
      neuralNet.setInputs(1, 0);
      neuralNet.setInputs(2, 1);
      neuralNet.setInputs(3, 1);
      neuralNet.run();
      exp = neuralNet.getOuputData(0);
      neuralNet.setDesiredOutput(0, 0);
      neuralNet.learn();
      if (exp > LowerThreshold)
      {
        return false;
      }
      
    
     

      return true;
    }


    private void TrainNetwork(int i1, int i2, int i3, int i4, int threadCount)
{
    NeuralNet neuralNetwork = threadCount == 1 ? this.NeuralNetwork : this.NeuralNetwork2;
    neuralNetwork.setInputs(0, i1);
    neuralNetwork.setInputs(1, i2);
    neuralNetwork.setInputs(2, i3);
    neuralNetwork.setInputs(3, i4);

    // If any input is 0, the desired output should be 0, else 1
    if (i1 == 0 || i2 == 0 || i3 == 0 || i4 == 0)
    {
        neuralNetwork.setDesiredOutput(0, 0);
        
    }
    else
    {
        neuralNetwork.setDesiredOutput(0, 1
            );
    }

    neuralNetwork.learn();
}

    



    private void OptimizeAccordinglyThread2()
    {
      bool breaking = false;
      HiddenLayersSoFar = 0;
      while (!breaking)
      {
        HiddenLayersSoFar++;
        label7.Invoke((MethodInvoker)(() => label7.Text = HiddenLayersSoFar.ToString()));

        NeuralNetwork2 = new NeuralNet(4, HiddenLayersSoFar, 1);
        int LocalMaxEpoch = MaxEpochs;

        for (TotalEpochs = 0; TotalEpochs <= LocalMaxEpoch; TotalEpochs++)
        {
          if (breaking) return;
          if (TotalEpochs % 56 == 0)
          {
            if (IsViable(NeuralNetwork2))
            {
                 
                  
                  button7.Text = "Use NN2";
                  n2 = true;
                  breaking = true;
                  MessageBox.Show($"Optimal solution found.");
                  return; // Exiting the function
                  
            }

            label6.Invoke((MethodInvoker)(() => label6.Text = TotalEpochs.ToString()));
          }

          PerformEpoch(2);

          // Check if breaking has been set to true in other parts of the code (e.g., from another thread)
          if (breaking)
          {
            return;
          }
        }
      }
    }


    private void OptimizeAccordingly()
    {
      bool breaking = false;
      HiddenLayersSoFar = 0;
      while (!breaking)
      {
        HiddenLayersSoFar += 1;
        hLayerCounter.Invoke((MethodInvoker)(() => hLayerCounter.Text = HiddenLayersSoFar.ToString()));

        NeuralNetwork = new NeuralNet(4, HiddenLayersSoFar, 1);
        int localTotalEpochs = 0;
        int localMaxEpochs = MaxEpochs;

        for (int tEpochs = 0; tEpochs <= localMaxEpochs; tEpochs++)
        {
          if (breaking) return;
          if (tEpochs % 43 == 0)
          {
            if (IsViable(NeuralNetwork))
            {
              
              button6.Text = "Use NN1";
              n1 = true;
              breaking = true;
              MessageBox.Show($"Optimal solution found.");
              return; // Exiting the function
              
                  
            }


            epochCounter.Invoke((MethodInvoker)(() => epochCounter.Text = tEpochs.ToString()));
          }

          PerformEpoch(1);
          localTotalEpochs++;

          // Check if breaking has been set to true in other parts of the code (e.g., from another thread)
          if (breaking)
          {
            return;
          }
        }
      }
    }




    public Form1()
    {
      InitializeComponent();

    }

    private void button1_Click(object sender, EventArgs e)
    {
      Thread t = new Thread(OptimizeAccordingly);
      t.Start();
      Thread t2 = new Thread(OptimizeAccordinglyThread2);
      t2.Start();
    }

    private void Form1_Load(object sender, EventArgs e)
    {

    }

    private void label12_Click(object sender, EventArgs e)
    {

    }

    private void button2_Click(object sender, EventArgs e)
    {
      if (button2.Text == "0")
        button2.Text = "1";
      else button2.Text = "0";
    }

    private void button3_Click(object sender, EventArgs e)
    {
      if (button3.Text == "0")
        button3.Text = "1";
      else button3.Text = "0";
    }

    private void button4_Click(object sender, EventArgs e)
    {
      if (button4.Text == "0")
        button4.Text = "1";
      else button4.Text = "0";
    }

    private void button5_Click(object sender, EventArgs e)
    {
     
      if (button5.Text == "0")
        button5.Text = "1";
      else button5.Text = "0";
    }

    private bool n1 = false;
    private bool n2 = false;
    private void button6_Click(object sender, EventArgs e)
    {
      if (!n1) return;
      var result = calculate(NeuralNetwork);
      textBox1.Text = result.ToString();
      textBox2.Text = discretize(result);
    }

    private string discretize(double result)
    {
      if (result > 0.5)
        return "1";
        return "0";
    }


  


  double calculate(NeuralNet net)
    {
      net.setInputs(0, Int32.Parse(button2.Text));
      net.setInputs(1, Int32.Parse(button3.Text));
      net.setInputs(2, Int32.Parse(button4.Text));
      net.setInputs(3, Int32.Parse(button5.Text));
      net.run();
      return net.getOuputData(0);
    }

  private void button7_Click(object sender, EventArgs e)
  {
    if (!n2) return;
    var result = calculate(NeuralNetwork2);
    textBox1.Text = result.ToString();
    textBox2.Text = discretize(result);
  }
  }
}
