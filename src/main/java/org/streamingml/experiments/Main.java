package org.streamingml.experiments;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws Exception {
        // run the algorithm
        int MAXCOEFF = 100;
        int length = 0, degree;
        double[] data = null;
        double[] coefficients;
        int i, method = - 1;

        // open the file
        String dataFileName = args[2];
        String line;
        int seriesPoints = 0;

        ClassLoader classLoader = Main.class.getClassLoader();

        if ((degree = Integer.parseInt(args[0])) >= MAXCOEFF) {
            System.out.printf("Maximum degree is %d\n", MAXCOEFF - 1);
            System.exit(- 1);
        }

        if (args[1].equals("mes")) { // max entropy spectrum
            method = 0;
        } else if (args[1].equals("lsq")) { // least squares yule walker
            method = 1;
        } else if (args[1].equals("rar")) { // recursive ar with kalman filter and forgetting
            method = 2;
        } else if (args[1].equals("rarma")) { // recursive arma with kalman filter and forgetting
            method = 3;
        } else if (args[1].equals("garima")) { // recursive gradient descent arima
            method = 4;
        } else if (args[1].equals("starma")) { // spatio temporal autocorrelation function
            method = 5;
        } else {
            System.out.printf("Didn't get a valid method\n");
            System.exit(- 2);
        }

        // SPATIO TEMPORAL ARMA MODELS
        if (method == 5) {
            try {
                // common border neighbours; two intersections are considered neighbours if they share a border
                String commBorderNeighborsFile = args[3];

                DMatrixRMaj dataSpaceTime = MatrixIO.loadCSV(classLoader.getResource(dataFileName).getFile(), true);
                SimpleMatrix dataSTAlgo = SimpleMatrix.wrap(dataSpaceTime);
                dataSpaceTime = MatrixIO.loadCSV(classLoader.getResource(commBorderNeighborsFile).getFile(), true);
                SimpleMatrix blistSTAlgo = SimpleMatrix.wrap(dataSpaceTime);
                SpatioTemporalARMAEstimator stARMAEstimator = new SpatioTemporalARMAEstimator(blistSTAlgo, dataSTAlgo.numCols(), degree, 2);
                double stCovCalc = stARMAEstimator.SpatioTemporalARMACovariance(dataSTAlgo, blistSTAlgo, degree - 1, degree - 1 , degree - 1);
                System.out.println("Model Component: STCOV");
                System.out.println(stCovCalc);
                System.out.println("---------------------");
                SimpleMatrix stACFCalc = stARMAEstimator.SpatioTemporalARMAAutocorrelationFunction(dataSTAlgo, blistSTAlgo, degree);
                System.out.println("Model Component: STACF");
                stACFCalc.print();
                System.out.println("---------------------");
                SimpleMatrix stPACFCalc = stARMAEstimator.SpatioTemporalARMAPartialAutocorrelationFunction(dataSTAlgo, blistSTAlgo, degree);
                System.out.println("Model Component: STPACF");
                stPACFCalc.print();
                System.out.println("---------------------");

                // sliding window execution
//                for (int dIter = 0; dIter < dataSTAlgo.numRows() - 2 * degree - 1; dIter++) {
//                    SimpleMatrix dataIn = dataSTAlgo.extractMatrix(dIter, 2*degree + dIter, 0, dataSTAlgo.numCols());
//                        stARMAEstimator.SpatioTemporalARMAIterate(dataIn);
//                }
                // dataset execution
                stARMAEstimator.SpatioTemporalARMAIterate(dataSTAlgo);
                for (int oid = 0; oid < stARMAEstimator.getKfOptmizer().getLearntModel().size(); oid ++){
                    String compId = "";
                    switch (oid){
                        case 0: compId = "AR COEF";
                            break;
                        case 1: compId = "STD AR COEF";
                            break;
                        case 2: compId = "MA COEF";
                            break;
                        case 3: compId = "STD MA COEF";
                            break;
                    }
                    if( oid <= 3) {
                        System.out.println("Model Component: " + compId);
                        stARMAEstimator.getKfOptmizer().getLearntModel().get(oid).print();
                        System.out.println("Model prediction");
                        stARMAEstimator.getKfOptmizer().getStarmaKfPrediction().print();
                        System.out.println("---------------------");
                    }
                }
                // calculate the LogLikelihood, AIC and BIC
                double LogLik = stARMAEstimator.SpatioTemporalARMACalculateLogLikelihood(dataSTAlgo, stARMAEstimator.getKfOptmizer().getLearntModel());
                System.out.println("Model LogLik: " + LogLik);
                double AIC = stARMAEstimator.SpatioTemporalARMACalculateAIC(
                        stARMAEstimator.SpatioTemporalARMACalculateLogLikelihood(dataSTAlgo, stARMAEstimator.getKfOptmizer().getLearntModel()),
                        stARMAEstimator.getKfOptmizer().getTimeOrder(),
                        stARMAEstimator.getKfOptmizer().getTimeOrder(),
                        dataSTAlgo.numRows());
                double BIC = stARMAEstimator.SpatioTemporalARMACalculateBIC(
                        stARMAEstimator.SpatioTemporalARMACalculateLogLikelihood(dataSTAlgo, stARMAEstimator.getKfOptmizer().getLearntModel()),
                        stARMAEstimator.getKfOptmizer().getTimeOrder(),
                        stARMAEstimator.getKfOptmizer().getTimeOrder(),
                        dataSTAlgo.numRows());
                System.out.println("Model AIC: " + AIC + "Model BIC: " + BIC);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else { // BASIC ARMA MODELS
            try {
                dataFileName = classLoader.getResource(dataFileName).getFile();
            } catch (NullPointerException e) {
                e.printStackTrace();
            }

            try {
                FileReader fileReaderCnt =
                        new FileReader(dataFileName);

                BufferedReader bufferedReaderCnt =
                        new BufferedReader(fileReaderCnt);

                while ((bufferedReaderCnt.readLine()) != null) {
                    seriesPoints++;
                }
                bufferedReaderCnt.close();

                data = new double[seriesPoints + 1];

                FileReader fileReader =
                        new FileReader(dataFileName);

                BufferedReader bufferedReader =
                        new BufferedReader(fileReader);

                while ((line = bufferedReader.readLine()) != null) {
                    data[length] = Double.parseDouble(line);
                    length++;
                }

                bufferedReader.close();
            } catch (FileNotFoundException ex) {
                System.out.println(
                        "Unable to open file '" +
                                dataFileName + "'");
                ex.printStackTrace();
            } catch (IOException ex) {
                ex.printStackTrace();
            }

            // instantiate and calculate and print the coefficients
            AutoRegressorEstimator ar = new AutoRegressorEstimator();
            ar.AutoRegressorEstimatorModel(data, length, degree, method);
            coefficients = ar.getSeriesCoef();
            if (method == 5) {
                System.out.println("Estimated (P)ACF:");
                for (i = 0; i < degree; i++) {
                    System.out.printf("%f\n", coefficients[i]);
                }
            } else if (method == 3) {
                System.out.println("Estimated ARMA coefficients:");
                for (i = 0; i <= 2 * degree + 1; i++) {
                    System.out.printf("%f\n", coefficients[i]);
                }
            }
            else if (method == 4) {
                System.out.println("Estimated ARIMA coefficients:");
                for (i = 0; i <= degree; i++) {
                    System.out.printf("%f\n", coefficients[i]);
                }
            }
            else {
                System.out.println("Estimated AR coefficients:");
                for (i = 0; i <= degree; i++) {
                    System.out.printf("%f\n", coefficients[i]);
                }
            }
            System.out.println("--------------------------");

            // get some stats on the series and the estimation
            int order = coefficients.length, j;
            double[] arcoeff = coefficients;
            double est;
            double rmserror = 0.0;

            // generate a series of the same length from the coefficients
            System.out.println("Original Time Series | Reconstructed Time Series");
            for (i = 0; i < length; i++) {
                est = 0.0;

                if (i > order) {
                    for (j = 0; j < order; j++) {
                        est += arcoeff[j] * data[i - j - 1];
                        rmserror += (est - data[i]) * (est - data[i]);
                    }
                }

                System.out.printf("%f | %f\n", data[i], est);
            }
            System.out.println("------------------------------------------------");
            System.out.println("Length of Time series | Order | RMSE");
            System.out.printf("%d | %d | %f\n", length - 1, order, Math.sqrt(rmserror / length));
            System.out.println("------------------------------------");
        }
    }

}
