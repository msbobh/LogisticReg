using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using System;
using Microsoft.VisualBasic.FileIO;
using System.IO;


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

        static public bool WriteCSV(string fname, double[] data)
        {

            StreamWriter outfile = null;
            try { outfile = new StreamWriter(fname); }
            catch (Exception e)
            {
                Console.WriteLine(strings.mystrings.File_Error, e);
                System.Environment.Exit(-1);
            }
            foreach (var row in data)
            {
                outfile.WriteLine(row.ToString());
            }
            outfile.Close();
            return true;
        }


            static public double CostFunc (Matrix<double> inMatrix, Matrix<double> Theta,Matrix<double> y, double lambda)
        {
            /* 
             * Computes the cost of using theta as the parameter for logistic regression to fit the data points in X and y
             */
            int m = inMatrix.RowCount;
            int cols = inMatrix.ColumnCount;
            double J = 0;
            double regterm;
            //Matrix<double> grad = Matrix<double>.Build.Dense(m, cols);
            // Vectorized 
            Matrix<double> z, hypothesis;
            z = inMatrix * Theta;
            // some issues here, shouldnt this be transpose?

            hypothesis = utilityfunctions.Sigmoid(z);

            regterm = utilityfunctions.L2Regularization(Theta, lambda, m);


             //Decomposition to check accuracy of primary equation
            Matrix<double> hypothesis1 = hypothesis.PointwiseLog();
            Matrix<double> term1 = -y.PointwiseMultiply(hypothesis1);
            Matrix<double> hypothesis2 = (1 - hypothesis).PointwiseLog();
            Matrix<double> term2 = (1 - y).PointwiseMultiply(hypothesis2);
            Matrix <double> foo = (term1 - term2);
            double foop = foo.RowSums().Sum();
            double foopfoop = foop + regterm;
            double foopity = 0.0084746;
            float fred = (1 / m);
            double foopJ = foopity * foopfoop;
            //Whew!
            
            
            /* J = (1/m) * sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))) + reg_term; */
            J = (1/ (double)m) * ((-y.PointwiseMultiply(hypothesis.PointwiseLog()) - ((1 - hypothesis).PointwiseLog()).PointwiseMultiply(1 - y)).RowSums().Sum()) + regterm;

            return J;
        }

        static public Matrix <double> GradientDescent(Matrix<double> X, Matrix<double> y, Matrix<double> Theta, double alpha, int iterations)
        {
            /* Performs Gradient Descent to learn Theta, updates theta by
             * taking num_iters gradient steps with learning rate alpha
             */

            double lambda = 10;
            double[] J_history = new double[iterations];
            J_history.Initialize();
            Matrix<double> error;
            
            int m = X.RowCount;
            for (int i = 0; i < iterations; i++)
            {
                error = (X * Theta) - y;
                J_history[i] = utilityfunctions.CostFunc(X, Theta,y,lambda);
                Theta = Theta - ((alpha / m) * X.Transpose() * error);
                Console.Write('.');
                if ( i % Console.WindowWidth -1 == 0) Console.WriteLine();
            }
            int x = 1;
            foreach (var cost in J_history)
            {
                Console.WriteLine("J:{0}", cost);
                x++;
            }
            utilityfunctions.WriteCSV("testcsv.csv", J_history);
            return Theta;
        }

        static public double L2Regularization (Matrix<double> Theta, double lambda, int m )
        {
            /* Regularization term excluding theta (0) Note both L1 and L2
             * regularization routines  are provided, comment/uncomment
             * as appropriate
             *          
             * L2 regularization uses the squared magnitude of coefficient
             * as penalty term to the loss function.
             */

            Theta = Theta.RemoveRow(0);
            Theta = Theta.PointwisePower(2);
            double foo = Theta.RowSums().Sum();
            double regterm = (lambda / (2 * m)) * (Theta.RowSums().Sum());
            return regterm;
        }

        static public double L1Regularization (Matrix<double> Theta, Double lambda, int m)
        {
            Theta = Theta.RemoveRow(0);
            Theta = Theta.PointwiseAbs();
            double regterm = (lambda / (2 * m)) * (Theta.RowSums().Sum());
            return regterm;
        }
    }
}

 