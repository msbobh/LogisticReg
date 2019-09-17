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
                // Constants
                //double lambda = 1;

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

                Matrix<double> g = Matrix<double>.Build.Dense(rows, cols);
                // calling sigmoid function = utilityfunctions.Sigmoid(input);

                /* Call gradient Descent Routine
                 * fprintf('\nRunning Gradient Descent ...\n')
                 * run gradient descent 
                 * theta = gradientDescent(X, y, theta, alpha, iterations)
                 */
                int iterations = 3500;
                double alpha = 0.01; // Alpha values to try .001, .003, .01, .03, .1, .3, 1 
                Console.WriteLine(mystrings.running, iterations);



                /* Investigate using ILNumerics unconstrained optimizaiton code, otherwise will need 
                 * to write a gradient descent routine. https://ilnumerics.net/unconstrained-optimization.html
                 * 
                 */

                // test code

                double lambda = 1;
                Matrix<double> theta = Functions.utilityfunctions.GradientDescent(input, labels, init_theta, alpha, iterations);

                // Use Delimited write to save the value of Theta
                // DelimitedWriter.Write("test.csv", g, ",");

            }
        }
    

}
