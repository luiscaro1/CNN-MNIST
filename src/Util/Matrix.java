package Util;

/*************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones collection of static methods for manipulating
 *  matrices.
 *
 *************************************************************************/

public class Matrix {

    // return C = A^T
    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[j][i] = A[i][j];
        return C;
    }

    public static double[][] transpose(double[] A) {
        double[][] result = new double[A.length][1];
        for (int i = 0; i < A.length; i++) {
            result[i][0] = A[i];
        }

        return result;

    }

    // return C = A + B
    public static double[][] add(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    // return C = A - B
    public static double[][] subtract(double[][] A, double[][] B) {
        int Am = A.length;
        int An = A[0].length;
        int Bm = B.length;
        int Bn = B[0].length;


        if (Am != Bm || An != Bn) {
            throw new IllegalArgumentException("The dimensions don't match!");
        }

        double[][] C = new double[Am][An];
        for (int i = 0; i < Am; i++)
            for (int j = 0; j < An; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    // return C = A * B
    public static double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }


    public static double[][] constant(double[][] matrix, double constant) {

        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[i][j] = matrix[i][j] * constant;
            }

        }
        return result;
    }


    public static double[][] oneToOne(double[][] M, double[][] N) {


        double[][] result = new double[M.length][M[0].length];


        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = M[i][j] * N[i][j];
            }
        }

        return result;

    }


}