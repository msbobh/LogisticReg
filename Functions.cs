using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using System;
using Microsoft.VisualBasic.FileIO;


namespace Functions
{
    class utilityfunctions
    {

        static public Matrix<double> Sigmoid(Matrix<double> input)
        {
            //g = 1 ./ (1 + exp (-z));
            for (int i = 0; i < input.RowCount; i++)
                for (int j = 0; j < input.ColumnCount; j++)
                {
                    input[i,j] = MathNet.Numerics.SpecialFunctions.Logistic(input[i, j]);
                }
             return input;
        }


        static public double CostFunc (Matrix<double> inMatrix, Matrix<double> Theta,Matrix<double> y, double lambda)
        {
            /* 
             * Computes the cost of using theta as the parameter for logistic regression to fit the data points in X and y
             */
            int m = inMatrix.RowCount;
            int cols = inMatrix.ColumnCount;
            double J = 0;
            //Matrix<double> grad = Matrix<double>.Build.Dense(m, cols);
            // Vectorized 
            Matrix<double> z, hypothesis;
            z = inMatrix * Theta;
            hypothesis = utilityfunctions.Sigmoid(z);

            /* Regularization term excluding theta (0) Note both L1 and L2
             * regularization routines  are provided, comment/uncomment
             * as appropriate
             */
            Theta.RemoveRow(0);
            Theta = Theta.PointwisePower(2);
            double regterm = (lambda / (2 * m)) * (Theta.RowSums().Sum());
            
            /* sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))) */

            J = ((-y.PointwiseMultiply(hypothesis.PointwiseLog()) - ((1 - hypothesis).PointwiseLog()).PointwiseMultiply(1 - y)).RowSums().Sum()) + regterm;

            /*Matrix<double> term1 = -y.PointwiseMultiply(hypothesis.PointwiseLog());
            Matrix<double> term2 = ((1 - hypothesis).PointwiseLog()).PointwiseMultiply(1-y);

            J = (term1 - term2).RowSums().Sum();
            */

            return J;
        }

        static public Matrix <double> GradientDescent(Matrix<double> X, Matrix<double> y, Matrix<double> Theta, double alpha, int iterations)
        {
            double[] J_history = new double[iterations];
            J_history.Initialize();
            Matrix<double> error;
            
            int m = X.RowCount;
            for (int i = 0; i < iterations; i++)
            {
                error = (X * Theta) - y;
                Theta = Theta - ((alpha / m) * X.Transpose() * error);
                J_history[i] = utilityfunctions.CostFunc(X, Theta,y,1);
                Console.Write('.');
                if ( i % Console.WindowWidth -1 == 0) Console.WriteLine();
            }
            int x = 1;
            foreach (var thingy in J_history)
            {
                Console.WriteLine("J:{0}", thingy);
                x++;
            }
            return Theta;
        }

        static public bool ValidateCSV(string path)
        {
                                   
            using (var parser = new TextFieldParser(path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");

                string[] line;
                while (!parser.EndOfData)
                {
                    try
                    {
                        line = parser.ReadFields();
                    }
                    catch (MalformedLineException ex)
                    {
                        Console.WriteLine("not a well formatted CSV");
                        return false;
                    }
                }
                return true;
            }
        }
    }
}

 