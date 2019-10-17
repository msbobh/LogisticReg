using System;
using ILNumerics;
using static ILNumerics.Globals;
using static ILNumerics.ILMath;


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
            int iterations = 1500;
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
             * var file = File.ReadAllText("csvread_sample34x24.csv");
                    Array<double> A = csvread<double>(file);

            // or read as single precision
                Array<float> S = csvread<float>(file);
             */
            var file = File.ReadAllText(trainingfile);
            Array<double> input = csvread<double>(file);
            file = File.ReadAllText(labelfile);
            Array<double> labels = csvread<double>(file);
            
            long rows = input.Size [0];
            long cols = input.Size [1];

                        
            if ((labels.Length != input.Length))
            {
                Console.WriteLine(mystrings.SamplesDontMatch, labels.Length, input.Length);
            }

            Console.WriteLine("Training Set rows = {0}, Columns = {1}", rows, cols);
            Console.WriteLine("Label set rows = {0}, Columns = {1}", labels.Size[0], labels.Size[1]);

            // Initial guess for Theta is all zeros and n x 1 vector of zeros, where n is the number of features (columns)

            Array <double> init_theta = zeros<double>(cols, 1);
            Array <double> ones_theta = ones <double>(cols, 1);
            Array<double> test_theta = rand (cols, 1); // Used this for testing chnages for ILNumerics
                        
            Console.WriteLine(mystrings.running, iterations);

            

            /* Investigate using ILNumerics unconstrained optimizaiton code, otherwise will need 
             * to write a gradient descent routine. https://ilnumerics.net/unconstrained-optimization.html
             * 
             */

                        
            Array <double> theta = Functions.utilityfunctions.GradientDescent(input, labels, init_theta, alpha, iterations, Lambda);

            StreamWriter checkthis;
            try
            {
                checkthis = new StreamWriter("LearnedTheta.csv");
                checkthis.Close();
            }
            catch (IOException)
            {
                Console.WriteLine("Oopsie");
            } 

            if (!utilityfunctions.FileOpen(ThetaFile))
            {
                Console.WriteLine(mystrings.File_in_use, ThetaFile);
                System.Environment.Exit(-1);
            }
            else
            {
                //csvwrite <double> (ThetaFile, theta);
            }
                
            
        }
    }






}
