package org.streamingml.experiments;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;

class SpatioTemporalKalmanFilterOptimizer {

    // Time series forecasting params
    private int nbAr;  // Number of AR parameters to be estimated
    private int nbMa;  // Number of MA parameters to be estimated
    private int dim; // Total number of parameters to be estimated
    private int timeOrder;   // max(maxTlagAr, maxTlagMa)

     // Initialize the Kalman filter
    private SimpleMatrix H;    // Observation matrix
    private SimpleMatrix ksi;   // Parameters vector (coefficients to estimate)
    private SimpleMatrix V;
    private SimpleMatrix noiseVar; // Variance of the white noise

    private SimpleMatrix W;
    private SimpleMatrix M;
    private SimpleMatrix nu;

    private ArrayList<SimpleMatrix> learntModel;

     // Rename and reshape
    private SimpleMatrix phi;
    private SimpleMatrix phiStd;
    private SimpleMatrix theta;
    private SimpleMatrix thetaStd;

    private SimpleMatrix arInd;
    private SimpleMatrix maInd;

    // Current prediction
    private SimpleMatrix starmaKfPrediction;

    // Build a configuration matrix, which coefficients one wants to estimate
    private SimpleMatrix SpatioTemporalParameterConfigurationBuilder(int tLag, int sLag) {
        SimpleMatrix out = new SimpleMatrix(tLag, sLag);
        for (int tId = 0; tId < tLag; tId++) {
            for (int sId = 0; sId < sLag; sId++) {
                out.set(tId, sId, 1.0);
            }
        }
        return out;
    }

    public SimpleMatrix getPhi() {
        return phi;
    }

    public SimpleMatrix getTheta() {
        return theta;
    }

    public int getTimeOrder() {
        return timeOrder;
    }

    public ArrayList<SimpleMatrix> getLearntModel() {
        return learntModel;
    }

    public SimpleMatrix getStarmaKfPrediction() {
        return starmaKfPrediction;
    }

    // Initialize the Kalman Filter Optimizer
    void SpatioTemporalKalmanFilterOptimizerInit(int timeSeriesSites, int tLag, int sLag)
    {
        int maxTlagAr;  // Max tlag of AR part
        int maxSlagAr;  // Max slag of AR part
        int maxTlagMa;  // Max tlag of MA part
        int maxSlagMa;  // Max slag of MA part

        // Build estimated coefficients config matrices
        SimpleMatrix arMat = SpatioTemporalParameterConfigurationBuilder(tLag, sLag);
        SimpleMatrix maMat = SpatioTemporalParameterConfigurationBuilder(tLag, sLag);

        // Create matrices containing the indices of the parameters to estimate
        nbAr = 0;
        for (int i = 0; i < arMat.numRows(); i++){
            for (int j = 0; j < arMat.numCols(); j++){
                if((int)arMat.get(i, j) != 0){
                    nbAr++;
                }
            }
        }
        nbMa = 0;
        for (int i = 0; i < maMat.numRows(); i++){
            for (int j = 0; j < maMat.numCols(); j++){
                if((int)maMat.get(i, j) != 0){
                    nbMa++;
                }
            }
        }

        // arInd contains the indices of the AR parameters to be estimated
        arInd = new SimpleMatrix(2, nbAr);
        int iter = 0;
        for (int tlag = 0; tlag < arMat.numRows(); tlag++) {
            for (int slag = 0; slag < arMat.numCols(); slag++) {
                if ((int)arMat.get(tlag, slag) != 0) {
                    arInd.set(0, iter, tlag);
                    arInd.set(1, iter, slag);
                    iter++;
                }
            }
        }

        // maInd contains the indices of the MA parameters to be estimated
        maInd = new SimpleMatrix(2, nbMa);
        iter = 0;
        for (int tlag = 0; tlag < maMat.numRows(); tlag++) {
            for (int slag = 0; slag < maMat.numCols(); slag++) {
                if ((int)maMat.get(tlag, slag)!= 0) {
                    maInd.set(0, iter, tlag);
                    maInd.set(1, iter, slag);
                    iter++;
                }
            }
        }

        // Parametrize the optimizer given the timeseries
        nbAr = arInd.numCols();
        nbMa = maInd.numCols();
        dim = nbAr + nbMa;
        maxTlagAr = (arInd.getNumElements() != 0) ? (int) ((arInd.extractVector(true, 0)).elementMaxAbs()) + 1 : 0;
        maxSlagAr = (arInd.getNumElements() != 0) ? (int) ((arInd.extractVector(true, 1)).elementMaxAbs()) + 1 : 0;
        maxTlagMa = (maInd.getNumElements() != 0) ? (int) ((maInd.extractVector(true, 0)).elementMaxAbs()) + 1 : 0;
        maxSlagMa = (maInd.getNumElements() != 0) ? (int) ((maInd.extractVector(true, 1)).elementMaxAbs()) + 1 : 0;
        timeOrder = (maxTlagAr > maxTlagMa) ? maxTlagAr : maxTlagMa;

        // Initialize the Kalman filter
        H = new SimpleMatrix(dim, timeSeriesSites);
        ksi = new SimpleMatrix(dim, 1);   // Unconditional mean of the prediction set to 0
        ksi.zero();
        V = SimpleMatrix.identity(dim).scale(1e5);
        noiseVar = SimpleMatrix.identity(timeSeriesSites).scale(1e-5); // Variance of the white noise set low

        W = new SimpleMatrix(timeSeriesSites, timeSeriesSites);
        M = new SimpleMatrix(timeSeriesSites, timeSeriesSites);
        nu = new SimpleMatrix(timeSeriesSites, 1);

        // Init the learntModel
        learntModel = new ArrayList<>();

        // Init the prediction
        starmaKfPrediction = new SimpleMatrix(1, timeSeriesSites);

        // Initialize the coefficients and their std vars
        phi = new SimpleMatrix(maxTlagAr, maxSlagAr); phi.zero();
        phiStd = new SimpleMatrix(maxTlagAr, maxSlagAr); phiStd.zero();
        theta = new SimpleMatrix(maxTlagMa, maxSlagMa); theta.zero();
        thetaStd = new SimpleMatrix(maxTlagMa, maxSlagMa); thetaStd.zero();
    }

    // Implmentation of a Kalman Filter Optimizer for STARMA Parameter Estimation
    // Based on: 'Study on Kalman filter in time series analysis' Cipra & Motyková (1987)
    //
    // The state space system considered is the following:
    // ksi[t+1] = ksi[t]
    // y[t] = H[t]' * ksi[t] + eps[t]
    //
    // where:
    // ksi is the vector of the parameters,
    // y is the vector of observations,
    // H is the matrix of past observations,
    // eps is the vector of innovation.
    void SpatioTemporalKalmanFilterIterateOptimizer(SimpleMatrix data, SimpleMatrix wlist, SimpleMatrix eps) {

        // Run the optimizer for the current window data
        for (int t = timeOrder; t < data.numRows(); t++) {

            // Update the observation matrix
            // Fill for 'phi', AR parameters
            for (int it = 0; it < nbAr; it++) {
                W.set(wlist.extractMatrix(0,
                        wlist.numRows(),
                        wlist.numRows() * ((int) arInd.get(1, it)),
                        wlist.numRows() * ((int) arInd.get(1, it) + 1)));
                double updateAr[] = ((data
                        .extractVector(true, t - 1 - (int) arInd.get(0, it)))
                        .mult(W.transpose()))
                        .getDDRM()
                        .data;
                H.setRow(it, 0, updateAr);
            }
            // Fill for 'theta', MA parameters
            for (int it = nbAr; it < dim; it++) {
                W.set(wlist.extractMatrix(0,
                        wlist.numRows(),
                        wlist.numRows() * ((int) maInd.get(1, it - nbAr)),
                        wlist.numRows() * ((int) maInd.get(1, it - nbAr) + 1)));
                double updateMa[] = ((eps
                        .extractVector(true, t - 1 - (int) maInd.get(0, it - nbAr)))
                        .mult(W.transpose()))
                        .getDDRM()
                        .data;
                H.setRow(it, 0, updateMa);
            }

            // Integrate the observations
            M = (((H.transpose().mult(V)).mult(H)).plus(SimpleMatrix.identity(data.numCols()))).invert();
            nu = (data.extractVector(true, t).transpose()).minus((H.transpose()).mult(ksi));

            // Prediction and update equations all-in-one
            ksi = ksi.plus(((V.mult(H)).mult(M)).mult(nu));
            V = V.minus((((V.mult(H)).mult(M)).mult(H.transpose())).mult(V));
            noiseVar = ((noiseVar.scale(t + 1 - timeOrder))
                    .plus((nu.mult(nu.transpose()))))
                    .divide((t + 2 - timeOrder));

            // Calculate the prediction
            starmaKfPrediction = (ksi.transpose()).mult(H);

            // Estimate the residuals
            eps.setRow(t, 0, (data.extractVector(true, t)
                    .minus((ksi.transpose()).mult(H)))
                    .getDDRM()
                    .data);

            // Get estimated std of the parameters
            SimpleMatrix sd = (V.diag().scale(noiseVar.trace())).divide(data.numCols());
            for (int sdi = 0; sdi < sd.numRows(); sdi++) {
                for (int sdj = 0; sdj < sd.numCols(); sdj++) {
                    sd.set(sdi, sdj, Math.sqrt(sd.get(sdi, sdj)));
                }
            }

            // Recover AR coefficients, phi and their std, phiStd
            for (int it = 0; it < nbAr; it++) {
                phi.set((int) arInd.get(0, it),
                        (int) arInd.get(1, it),
                        ksi.get(it, 0));
                phiStd.set((int) arInd.get(0, it),
                        (int) arInd.get(1, it),
                        sd.get(it, 0));
            }

            // Recover MA coefficients, theta and their std, thetaStd
            for (int it = nbAr; it < dim; it++) {
                theta.set((int) maInd.get(0, it - nbAr),
                        (int) maInd.get(1, it - nbAr),
                        ksi.get(it, 0));
                thetaStd.set((int) maInd.get(0, it - nbAr),
                        (int) maInd.get(1, it - nbAr),
                        sd.get(it, 0));
            }
        }
        // Update the learntModel with latest estimates for the current window
        learntModel = new ArrayList<>();
        learntModel.add(phi);
        learntModel.add(phiStd);
        learntModel.add(theta);
        learntModel.add(thetaStd);
        learntModel.add(noiseVar);

        // Calculate as well the estimated noise variance
        SimpleMatrix noiseVarCalc = new SimpleMatrix(1, 1);
        noiseVarCalc.set(0,
                0,
                (noiseVar.diag()).elementSum() / noiseVar.numCols());
        learntModel.add(noiseVarCalc);
    }
}

public class SpatioTemporalARMAEstimator {

    // Current spatial weights list
    private SimpleMatrix wlist;
    // Time lag and spatial lag of the timeseries
    private int tLag;
    private int sLag;
    // The number of spatially correlated timeseries
    private int timeseriesSites;
    private SpatioTemporalKalmanFilterOptimizer kfOptmizer;

    public SpatioTemporalARMAEstimator(SimpleMatrix w, int sites, int t, int s){
        wlist = w; tLag = t; sLag = s; timeseriesSites = sites;
        kfOptmizer = new SpatioTemporalKalmanFilterOptimizer();
        kfOptmizer.SpatioTemporalKalmanFilterOptimizerInit(sites, t, s);
    }

    public SpatioTemporalKalmanFilterOptimizer getKfOptmizer() {
        return kfOptmizer;
    }

    // Iteratively computes the residuals of a given starma learntModel

    // Inputs:
    //  data: A matrix object containing the space time process observations
    //	wlist: The list of weight matrices, first one being identity
    //
    // Returns:
    //  eps: residuals of the learntModel

    private SimpleMatrix SpatioTemporalARMACalculateResiduals(SimpleMatrix data) {

        SimpleMatrix phi = kfOptmizer.getPhi();
        SimpleMatrix theta = kfOptmizer.getTheta();
        SimpleMatrix eps = data.copy();	// Will hold the residuals
        SimpleMatrix W = new SimpleMatrix(data.numCols(), data.numCols());
        int tlim;

        // Remove AR part iteratively
        for (int t = 0; t < data.numRows(); t++) {
            tlim = (t < phi.numRows()) ? t : phi.numRows();
            for (int tlag = 0; tlag < tlim; tlag++) {
                for (int slag = 0; slag < phi.numCols(); slag++) {
                    W.set(wlist.extractMatrix(0, wlist.numRows(), wlist.numRows() * slag, wlist.numRows() * (slag  + 1)));
                    double rowDataEps[] =  (eps.extractVector(true, t)
                            .minus((((data.extractVector(true, t - tlag - 1))
                                    .mult(W.transpose())))
                                    .scale(phi.get(tlag, slag))))
                            .getDDRM().data;
                    eps.setRow(t, 0, rowDataEps);
                }
            }
        }

        // Remove MA part iteratively
        for (int t = 0; t < data.numRows(); t++) {
            tlim = (t < theta.numRows()) ? t : theta.numRows();
            for (int tlag = 0; tlag < tlim; tlag++) {
                for (int slag = 0; slag < theta.numCols(); slag++) {
                    W.set(wlist.extractMatrix(0, wlist.numRows(), wlist.numRows() * slag, wlist.numRows() * (slag  + 1)));
                    double rowDataEps[] =  (eps.extractVector(true, t)
                            .minus((eps.extractVector(true, t - tlag - 1)
                                    .mult(W.transpose()))
                                    .scale(theta.get(tlag, slag))))
                            .getDDRM().data;
                    eps.setRow(t, 0, rowDataEps);
                }
            }
        }

        return eps;

    }

    // Implementations of Space-Time ARMA models an specific utilities
    // It follows the three-stage iterative learntModel building procedure extended to space-time modelling by (Pfeifer and Deutsch, 1980).
    //    The three stages of the iterative learntModel building procedure are as follow, after centering the spacetime
    //    series:
    //            - Identification: Using ACF and PACF identify which parameters should be estimated.
    //            - Estimation: Use Kalman Filter to estimate the starmaparameters.
    //            - Diagnostic: Check whether the residuals of the models are similar to white noise.

    SimpleMatrix SpatioTemporalARMADataCenteringScaling(SimpleMatrix data) {
        // Center and scale a data frame such that its mean is 0 and sd is 1.

        // center the data
        double dataMean = data.elementSum() / (data.numCols() * data.numRows());
        data = data.minus(dataMean);

        // scale the data
        double dataStd = Math.sqrt(((data.mult(data)).elementSum()) / ( data.numRows() * data.numCols() - 1.0));
        data = data.divide(dataStd);

        return data;
    }

    // Implements the space-time covariance calculation of the series data between
    // space slag1-th and space slag2-th order neighbours at time lag tlag.
    //
    // Used as an internal function for the computation of the STACF and STPACF slag1 and slag2 must be lower than length(wlist).
    //
    // data - a matrix or data frame containing the space-time series: row-wise should be the
    // temporal observations, with each column corresponding to a site.
    //
    // wlist - a list of the weight matrices for each k-th order neighbours, first one being the identity.
    //
    // slag - the space lags for the space-time covariance.
    //
    // tlag - the time lag for the space-time covariance.
    double SpatioTemporalARMACovariance(SimpleMatrix data, SimpleMatrix wlist, int slag1, int slag2, int tlag) {
        // The function is defined as, per (Pfeifer & Stuart, 1980), using the following formula:
        // cov(maxSlagAr,k,s) = E[ (W(maxSlagAr)*z(t))' * (W(k)*z(t+s)) ] / N = Tr( W(k)'*W(maxSlagAr) * E[z(t)*z(t+s)'] ) / N
        double out = 0;
        int indLim = data.numRows() - tlag;

        SimpleMatrix wK = new SimpleMatrix(wlist.numRows(), wlist.numRows());
        wK.set(wlist.extractMatrix(0, wlist.numRows(), wlist.numRows() * slag1, wlist.numRows() * (slag1  + 1)));
        SimpleMatrix wL = new SimpleMatrix(1, wlist.numCols());
        wL.set(wlist.extractMatrix(0, wlist.numRows(), wlist.numRows() * slag2, wlist.numRows() * (slag2  + 1)));
        SimpleMatrix pW = wL.transpose().mult(wK);

        for (int it = 0; it < indLim; it++) {
            SimpleMatrix slidePre = data.extractVector(true, it).transpose();
            SimpleMatrix slidePost = data.extractVector(true, it + tlag);
            out += (pW.mult(slidePre).mult(slidePost)).trace();
        }

        out /= indLim * data.numCols();

        return out;
    }

    // Implements the calculation of the Space Time Covariance Matrix
    SimpleMatrix SpatioTemporalARMACovarianceMatrix(SimpleMatrix data, SimpleMatrix wlist, int tlag) {

        int slagMax = wlist.numCols()/ wlist.numRows();
        SimpleMatrix out = new SimpleMatrix(slagMax, slagMax);

        for (int i = 0; i < slagMax; i++) {
            for (int j = 0; j < slagMax; j++) {
                out.set(i, j, SpatioTemporalARMACovariance(data, wlist, i, j, tlag));
            }
        }

        return out;

    }

    // Implements the calculation of the Space Time Covariance Vector
    SimpleMatrix SpatioTemporalARMACovarianceVector(SimpleMatrix data, SimpleMatrix wlist, int tlagMax) {

        int slagMax = wlist.numCols()/ wlist.numRows();
        SimpleMatrix out = new SimpleMatrix(slagMax * tlagMax, 1);

        for (int tlag = 1; tlag <= tlagMax; tlag++) {
            for (int slag = 0; slag < slagMax; slag++) {
                out.set((tlag - 1) * slagMax + slag, SpatioTemporalARMACovariance(data, wlist, slag, 0, tlag));
            }
        }

        return out;

    }

    // Implements the calculation of the Space Time Covariance Matrix generalization on all time lags
    SimpleMatrix SpatioTemporalARMACovarianceMatrixExtended(SimpleMatrix data, SimpleMatrix wlist, int tlagMax) {

        int slagMax = wlist.numCols()/wlist.numRows();
        SimpleMatrix slideye = new SimpleMatrix(tlagMax, 2*tlagMax - 1);
        slideye.zero();
        for (int i = 0; i < tlagMax; i++){
            for (int j = 0; j < 2*tlagMax - 1; j++){
                if(i == j) {
                    slideye.set(i, j, 1.0);
                }
            }
        }
        SimpleMatrix out = new SimpleMatrix(slagMax * tlagMax, slagMax * tlagMax);
        out.zero();

        for (int tlag = 1; tlag < tlagMax; tlag++) {
            out = out.plus((slideye.extractMatrix(0, tlagMax, tlag,  tlag + tlagMax))
                    .kron(SpatioTemporalARMACovarianceMatrix(data, wlist, tlag)));
        }

        out = out.plus(out.transpose());
        out = out.plus(SimpleMatrix.identity(tlagMax).kron(SpatioTemporalARMACovarianceMatrix(data, wlist, 0)));

        return out;

    }

    // The function implements one of the main tools for the identification and the diagnostic part of the
    // iterative space time ARIMA. Compute the autocorrelation function (ACF) of a space-time series.
    //
    // Compute the space-time autocorrelation of the serie data between s-th and 0-th order neighbors at time lag t,
    // for space lags ranging from 0 to length(wlist) and t ranging from 1 to tlag.max.
    SimpleMatrix SpatioTemporalARMAAutocorrelationFunction(SimpleMatrix data, SimpleMatrix wlist, int tlagMax) {

        SimpleMatrix out = new SimpleMatrix(tlagMax, wlist.numCols()/wlist.numRows());

        double covNoSTLag = SpatioTemporalARMACovariance(data, wlist, 0, 0, 0);
        for (int slag = 0; slag < (wlist.numCols()/wlist.numRows()); slag++) {
            double covSNoTLag = SpatioTemporalARMACovariance(data, wlist, slag, slag, 0);
            for (int tlag = 1; tlag <= tlagMax; tlag++) {
                double covTNoSLag = SpatioTemporalARMACovariance(data, wlist, slag, 0, tlag);
                out.set(tlag - 1, slag, covTNoSLag / Math.sqrt( covSNoTLag * covNoSTLag ));
            }
        }

        return out;
    }

    // The function implements one of the main tools for the identification and the diagnostic part of the
    // iterative space time ARIMA. Compute the partial autocorrelation function (PACF) of a space-time series.
    //
//     The PACFs are computed solving iteratively the Yule Walker equations for
//     increasing time lags and space lags. Note that the identification might be biased if the partial
//     autocorrelation functions are not computed with enough space lags, since Yule Walker equations
//     are sensitive to the maximum space lag given.
    SimpleMatrix SpatioTemporalARMAPartialAutocorrelationFunction(SimpleMatrix data, SimpleMatrix wlist, int tlagMax) {

        int slagMax = wlist.numCols()/wlist.numRows();
        SimpleMatrix yWmat = SpatioTemporalARMACovarianceMatrixExtended(data, wlist, tlagMax);
        SimpleMatrix yWvec = SpatioTemporalARMACovarianceVector(data, wlist, tlagMax);
        SimpleMatrix out = new SimpleMatrix(tlagMax, slagMax);

        for (int tlag = 1; tlag <= tlagMax; tlag++) {
            for (int slag = 0; slag < slagMax; slag++) {
                int index = (tlag - 1) * slagMax + slag;
                SimpleMatrix A = yWmat.extractMatrix(0, index + 1, 0, index + 1);
                SimpleMatrix b = yWvec.extractMatrix(0, index + 1, 0, 1);
                SimpleMatrix sol = A.solve(b);
                out.set(tlag - 1, slag, sol.get(index));
            }
        }

        return out;

    }

    // starma parameter estimation function based on a Kalman filter optimizer

    //  Inputs:
    //   - data: A matrix object containing the space time process observations
    //
    //	Updates the learntModel, a list containing the following elements:
    //	- phi: The estimated AR parameters
    //	- phiStd: The estimated standard error of the AR parameters
    //	- theta: The estimated MA parameters
    //	- thetaStd: The estimated standard error of the MA parameters
    //	- noiseVar: The estimated white noise variance matrix
    //
    void SpatioTemporalARMAIterate(SimpleMatrix dataWin) {

        // Optimizer iterations, balance between accuracy and speed
        int iterate = 5;

        // Estimated residuals
        SimpleMatrix eps = dataWin.copy();

        // First iteration for AR coefficients estimation
        this.kfOptmizer.SpatioTemporalKalmanFilterIterateOptimizer(dataWin, wlist, eps);
        SimpleMatrix rowDataResiduals = new SimpleMatrix(this.kfOptmizer.getTimeOrder(), dataWin.numCols());
        // Update also MA coefficients
        while ((iterate--) != 0) {
            // Continuously estimate also the residuals on the current event window
            for (int rid = 0; rid < this.kfOptmizer.getTimeOrder(); rid++) {
                rowDataResiduals = SpatioTemporalARMACalculateResiduals(dataWin
                        .extractMatrix(0, rid, 0, dataWin.numCols()));
            }
            for (int rid = 0; rid < rowDataResiduals.numRows(); rid++) {
                eps.setRow(rid, 0, rowDataResiduals.extractVector(true, rid).getDDRM().data);
            }
            // Update the learntModel for the current window with both AR and MA
            this.kfOptmizer.SpatioTemporalKalmanFilterIterateOptimizer(dataWin, wlist, eps);
        }
        // add residuals in the model
        this.kfOptmizer.getLearntModel().add(SpatioTemporalARMACalculateResiduals(dataWin));
    }

    // Computes the conditional log likelihood of a given starma learntModel.

    // Inputs:
    //    - data: A matrix object containing the space time process observations
    //	- learntModel: The parameters of the learntModel ; this argument should be the output of the 'starmaCPP' function
    //
    // Returns:
    //    The conditional log likelihood of the starma learntModel.

    double SpatioTemporalARMACalculateLogLikelihood(SimpleMatrix data, ArrayList<SimpleMatrix> learntModel) {

        SimpleMatrix residuals = learntModel.get(6);
        double noiseVar = ((learntModel.get(4)).trace()) / data.numCols();
        double L = data.numCols() * data.numRows() * (Math.log(2 * Math.PI) + Math.log(noiseVar));

        for (int t = 0; t < data.numRows(); t++) {
            double resUpdate = ((residuals.extractVector(true, t)
                    .divide(noiseVar)
                    .mult((residuals.extractVector(true, t)).transpose())))
                    .get(0,0);
            L += resUpdate;
        }

        return -L/2.0;

    }

    // Computes the Bayesian Information Criterion (BIC) of a given starma learntModel.

    // Inputs:
    //  - loglik: The conditional log likelihood of the starma learntModel.
    //  - ar: AR order
    //  - ma: MA order
    //  - n: size of data

    // Returns:
    //    The BIC of the starma learntModel.

    double SpatioTemporalARMACalculateBIC(double loglik, int ar, int ma, int n) {

        // BIC = -2 * LogLik + log(n) * (ar + ma);
        return -2.0 * loglik + Math.log(n) * (ar + ma);

    }

    // Computes the Akaike Information Criterion (AIC) of a given starma learntModel.

    // Inputs:
    //  - loglik: The conditional log likelihood of the starma learntModel.
    //  - ar: AR order
    //  - ma: MA order
    //  - n: size of data

    // Returns:
    //    The AIC of the starma learntModel.

    double SpatioTemporalARMACalculateAIC(double loglik, int ar, int ma, int n) {

        // AIC = -2 * LogLik + 2 * (ar + ma)
        return -2.0 * loglik + 2.0 * (ar + ma);

    }

}
