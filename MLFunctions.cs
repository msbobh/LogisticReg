using ILNumerics;
using static ILNumerics.Globals;
using static ILNumerics.ILMath;
using System;

namespace MLFunctions
{ 
    class math
    { 
        static public RetArray <double> Sigmoid(Array <double> input)
        {
            //g = 1 ./ (1 + exp (-z));
            for (int i = 0; i < input.S[0]; i++)
                for (int j = 0; j < input.S[1]; j++)
                {
                    input[i, j] = logistic(input[i, j]);
                }
            return input;
        }

        static public RetArray <double> Siggy (InArray <double> input)
        {
            // g = 1 / (1 + exp (-z))
            return logistic(input);
        }

        static public double L2Regularization(Array <double> Theta, double lambda, long m)
        {
            // Regularization term excluding theta (0) Note both L1 and L2
            // regularization routines  are provided, comment/uncomment
            // as appropriate
            //           
            // L2 regularization uses the squared magnitude of coefficient
            // as penalty term to the loss function.
            // reg_term = (lambda/(2*m))* sum (theta(2:end).^2);
            // 
                      

            Theta = Theta * Theta;    // Take the square of the Theta's
            double regterm = (lambda / (2 * m)) * sum(Theta[r(1, end), r(0, end)], 0).GetValue(0, 0);
            return regterm;
         }

        static public double L1Regularization(InArray <double> Theta, Double lambda, long m)
        {
            Theta = abs (Theta[r(1,end), full]); // strip off first row (one) and take the absolute value
            double regterm = (lambda / (2 * m)) * sum (Theta).GetValue(0,0); 
            return regterm;
        }

        static public double CostFunc(InArray <double> inMatrix, InArray <double> Theta, InArray <double> y, double lambda)
        {
            // 
            // Computes the cost of using theta as the parameter for logistic regression to fit the data points in X and y
            // 
            long m = inMatrix.S[0];
            long cols = inMatrix.S[1]; 
            double J = 0;
            double regterm;
            //Matrix<double> grad = Matrix<double>.Build.Dense(m, cols);
            // Vectorized 
            Array <double> z, hypothesis;
            z = ILMath.multiply (inMatrix, Theta);
            // some issues here, shouldnt this be transpose?

            hypothesis = MLFunctions.math.Sigmoid(z);
            Array<double> testhypothesis = MLFunctions.math.Siggy(z);
            Console.WriteLine("interitive sigmoid {0}", hypothesis.ToString());
            Console.WriteLine("Vectorized sigmoid, {0}", testhypothesis.ToString());
            Console.ReadKey();
            regterm = MLFunctions.math.L2Regularization(Theta, lambda, m);


            //
            // Decomposition to check accuracy of primary equation
            // Matrix<double> hypothesis1 = hypothesis.PointwiseLog();
            // Matrix<double> term1 = -y.PointwiseMultiply(hypothesis1);
            // Matrix<double> hypothesis2 = (1 - hypothesis).PointwiseLog();
            // Matrix<double> term2 = (1 - y).PointwiseMultiply(hypothesis2);
            // Matrix <double> foo = (term1 - term2);
            // double foop = foo.RowSums().Sum();
            // double foopfoop = foop + regterm;
            // double foopity = 0.0084746;
            // float fred = (1 / m);
            // double foopJ = foopity * foopfoop;
            // 
            // Whew!
            //
            Array<double> term1 = -y * log(hypothesis);
            Array<double> term2 = (1 - y) * log(1 - hypothesis);
            double term3 = ((1 / (double)m) * sum(term1 - term2)).GetValue(0,0);

            // J = (1/m) * sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))) + reg_term; 
            //J = (1 / (double)m) * ((-y.PointwiseMultiply(hypothesis.PointwiseLog()) - ((1 - hypothesis).PointwiseLog()).PointwiseMultiply(1 - y)).RowSums().Sum()) + regterm;

            return term3 ;
        } 
    } 
}