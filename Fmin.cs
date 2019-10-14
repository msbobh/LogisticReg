using System;
using MathNet.Numerics.LinearAlgebra;

namespace minimize
{
    class fmin
    {

        public static void function_grad(Matrix <double> X, Matrix <double> theta,Matrix <double> y, int m,double lambda, ref double func,double[] grad/* , object obj*/)
        {
            Matrix<double> z, hypothesis;
            double regterm,J;
            z = X * theta;
            hypothesis = MLFunctions.math.Sigmoid(z);

            regterm = MLFunctions.math.L2Regularization(theta, lambda, m);
            // this callback calculates f(x,theta) = (1/m) * sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))) + reg_term;
            // and its derivatives df/dx and df/dxtheta
            
            J = (1 / (double)m) * ((-y.PointwiseMultiply(hypothesis.PointwiseLog()) - ((1 - hypothesis).PointwiseLog()).PointwiseMultiply(1 - y)).RowSums().Sum()) + regterm;

            func = J; // Return the error
            // grad(1) = (1/m) * (X(:,1)'*(hypothesis-y));
            // grad(2:end) = (1 / m) * (X(:, 2:end)'*(hypothesis-y)) + (lambda/m) * (theta(2:end));


            Vector <double> grad1 = (1 / (double)m) * ((hypothesis - y).Transpose()).Multiply(X.Column(0));
            grad[0] = grad1[0];
            Matrix<double> thetaminus1 = theta.RemoveRow(0);
            
            Matrix<double> Xless1 = X.RemoveColumn(0);
            Matrix<double> term1 = Xless1.Transpose() * (hypothesis - y);
            Matrix<double> term2 = (lambda / (double)m) * thetaminus1;
            Matrix<double> grad2 = (1 / (double)m) * term1 + term2;
            grad  = grad2.AsColumnMajorArray();
            
        }

       /* internal static void function_grad(Matrix<double> input, Matrix<double> init_theta, Matrix<double> labels, int rows, double lambda, double foo)
        {
            throw new NotImplementedException();
        }*/
    }
}

