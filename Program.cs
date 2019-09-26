using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.Data.Text;
using Functions;
using System.IO;
using strings;

namespace LogisticReg
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initializations
            double alpha = .01; // Alpha values to try .001, .003, .01, .03, .1, .3, 1 
            double Lambda = 10;
            int iterations = 500;
            string ThetaFile = "Thetasave.csv";

            if (args.Length < 2)
            {
                Console.WriteLine(mystrings.usage);
                System.Environment.Exit(-1);
            }
            if (!File.Exists(args[0]))
            {
                Console.WriteLine("Training file {0} not found!", args[0]);
                System.Environment.Exit(-1);
            }

            if (!File.Exists(args[1]))
            {
                Console.WriteLine("Label file {0} not found!", args[1]);
                System.Environment.Exit(-1);
            }

            if (args.Length > 2 && File.Exists(args[2]))
            {
                alpha = Convert.ToDouble(args[2]); ;

            }
            else
            {
                Console.WriteLine ("Using default alpha {0}", alpha);
            }

            string trainingfile = args[0];
            string labelfile = args[1];
            // maybe add check for csv format??

            /*
             * Delimited Reader Param description
             * Delimited Reader (only Single, Double, Complex and COmplex32)
             * second param Sparse (True) or dense matrix (false)
             * THird Param = delimeter
             * Fourth has headers (T|F)
             */


            Matrix<double> input = DelimitedReader.Read<double>(trainingfile, false, ",", false);
            Matrix<double> labels = DelimitedReader.Read<double>(labelfile, false, "'", false);
            if ((labels.RowCount != input.RowCount))
            {
                Console.WriteLine(mystrings.SamplesDontMatch, labels.RowCount, input.RowCount);
            }

            int rows = input.RowCount;
            int cols = input.ColumnCount;
            Console.WriteLine("Training Set rows = {0}, Columns = {1}", rows, cols);
            Console.WriteLine("Lable set rows = {0}, Columns = {1}", labels.RowCount, labels.ColumnCount);

            // Initial guess for Theta is all zeros and n x 1 vector of zeros, where n is the number of features (columns)

            Matrix<double> init_theta = Matrix<double>.Build.Dense(cols, 1);
            Matrix<double> ones_theta = Matrix<double>.Build.Dense(cols, 1, 1);
                        
            Console.WriteLine(mystrings.running, iterations);



            /* Investigate using ILNumerics unconstrained optimizaiton code, otherwise will need 
             * to write a gradient descent routine. https://ilnumerics.net/unconstrained-optimization.html
             * 
             */


            
            Matrix<double> theta = Functions.utilityfunctions.GradientDescent(input, labels, ones_theta, alpha, iterations, Lambda);

            /* StreamWriter checkthis;
            try
            {
                checkthis = new StreamWriter("LearnedTheta.csv");
                checkthis.Close();
            }
            catch (IOException)
            {
                Console.WriteLine("Oopsie");
            } */

            if (!utilityfunctions.FileOpen(ThetaFile))
            {
                Console.WriteLine(mystrings.File_in_use, ThetaFile);
                System.Environment.Exit(-1);
            }
            else
            {
                DelimitedWriter.Write(ThetaFile, theta, ",");
            }
                
            
            // Use Delimited write to save the value of Theta
            // DelimitedWriter.Write("test.csv", g, ",");

        }
    }






}
