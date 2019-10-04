using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using System;
using Microsoft.VisualBasic.FileIO;
using System.IO;


namespace Functions
{
    class utilityfunctions
    {

        static bool IsFileinUse(FileInfo file)
        {
            FileStream stream = null;

            try
            {
                stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                //the file is unavailable because it is:
                //still being written to
                //or being processed by another thread
                //or does not exist (has already been processed)
                return true;
            }
            finally
            {
                if (stream != null)
                    stream.Close();
            }
            return false;
        }

        static public bool FileOpen(string chkfilename)
        {
            StreamWriter checkthis;
            try
            {
                checkthis = new StreamWriter(chkfilename);
                checkthis.Close();
            }
            catch (IOException)
            {
                return false;
            }
            return true;
        }


        static public bool WriteCSV(string CSVname, double[] jValues)
        {
         
            StreamWriter outfile = null;
            try
            {
                outfile = new StreamWriter(CSVname);
            }
            catch (IOException)
            {
                return false;
            }
            
            foreach (var row in jValues)
            {
                outfile.WriteLine(row.ToString());
            }

            outfile.Close();
            return true;
        }

        
        static public Matrix <double> GradientDescent(Matrix<double> X, Matrix<double> y, Matrix<double> Theta, double alpha, int iterations, double lambda)
        {
            /* Performs Gradient Descent to learn Theta, updates theta by
             * taking num_iters gradient steps with learning rate alpha
             */

            double[] J_history = new double[iterations];
            J_history.Initialize();
            Matrix<double> error;
            string JValsFname = "JValues.csv";
            
            int m = X.RowCount;
            for (int i = 0; i < iterations; i++)
            {
                error = (X * Theta) - y;
                Theta = Theta - ((alpha / m) * X.Transpose() * error);
                Console.Write('.');
                if ( i % Console.WindowWidth -1 == 0) Console.WriteLine();
                J_history[i] = MLFunctions.math.CostFunc(X, Theta, y, lambda);
            }
            int x = 1;
            foreach (var cost in J_history)
            {
                Console.WriteLine("J = " + string.Format("{0:0.0000}", cost));
                x++;
            }
            if ((utilityfunctions.WriteCSV(JValsFname, J_history) == false))
            {
                Console.WriteLine(strings.mystrings.File_in_use, JValsFname);
                System.Environment.Exit(-1);
            }
                
            return Theta;
        }

        
    }
}

 