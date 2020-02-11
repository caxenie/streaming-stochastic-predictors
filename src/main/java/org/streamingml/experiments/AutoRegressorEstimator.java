package org.streamingml.experiments;

import org.ejml.UtilEjml;
import org.ejml.data.Complex_F64;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.EigenDecomposition_F64;
import org.ejml.ops.ComplexMath_F64;
import org.ejml.simple.SimpleMatrix;

class AutoRegressorEstimator {

    private int mode;
    private double[] inputSeries;
    private int seriesLen;
    // There is no straightforward way to determine the correct model order. As one increases the order of
    // the model the root mean square RMS error generally decreases quickly up to some order and then
    // more slowly. An order just after the point at which the RMS error flattens out is usually an
    // appropriate order. There are more formal techniques for choosing the model order, the most
    // common of which is the Akaike Information Criterion.
    private int seriesDeg;
    private double[] seriesCoef;

    private void setInputSeries(double[] inputSeries) {
        this.inputSeries = inputSeries;
    }

    private double[] getInputSeries() {
        return inputSeries;
    }

    private  void setSeriesLen(int seriesLen) {
        this.seriesLen = seriesLen;
    }

    private  int getSeriesLen() {
        return seriesLen;
    }

    private void setSeriesDeg(int seriesDeg) {
        this.seriesDeg = seriesDeg;
    }

    private int getSeriesDeg() {
        return seriesDeg;
    }

    private void setMode(int mode) {
        this.mode = mode;
    }

    private void setSeriesCoef(double[] seriesCoef) {
        this.seriesCoef = seriesCoef;
    }

    double[] getSeriesCoef() {
        return seriesCoef;
    }

    void AutoRegressorEstimatorModel(double[] s, int l, int d, int m) {

        setMode(m);
        setInputSeries(s);
        setSeriesLen(l);
        setSeriesDeg(d);
        SimpleMatrix wlist;
        int i, t;
        double mean = 0.0;

        double[] coeff = new double[getSeriesDeg() + 1];

        double[] w = new double[getSeriesLen()];
        double[] h = new double[getSeriesDeg() + 1];
        double[] g = new double[getSeriesDeg() + 2];

        double[] per = new double[getSeriesLen() + 1];
        double[] pef = new double[getSeriesLen() + 1];

        double[][] ar = new double[getSeriesDeg() + 1][getSeriesDeg() + 1]; // AR coefficients, all degrees

        // determine the mean and subtract the mean from the input series
//        for (t = 0; t < getSeriesLen(); t++) {
//            mean += getInputSeries()[t];
//        }
//
//        mean /= (double) getSeriesLen();
//
//        for (t = 0; t < getSeriesLen(); t++) {
//            w[t] = getInputSeries()[t] - mean;
//        }
//

        for (t = 0; t < getSeriesLen(); t++) {
            w[t] = getInputSeries()[t];
        }

        // perform the selected AR calculation
        switch (mode){
            case 0:// MAX_ENTROPY
                ARMaxEntropy(w, getSeriesLen(), getSeriesDeg(), ar, per, pef, h, g);
                for (i = 1; i <= getSeriesDeg(); i++) {
                    coeff[i - 1] = - ar[getSeriesDeg()][i];
                }
                setSeriesCoef(coeff);
                break;
            case 1: // LEAST_SQUARES
                if (getSeriesCoef() == null) {
                    setSeriesCoef(new double[getSeriesDeg() + 1]);
                }
                ARLestSquares(w, getSeriesLen(), getSeriesDeg(), getSeriesCoef());
                break;
            case 2: // Recursive AR
                if (getSeriesCoef() == null) {
                    setSeriesCoef(new double[getSeriesDeg() + 1]);
                }
                ARKalmanRecursive(w,getSeriesLen(),getSeriesDeg(), 0.992, getSeriesCoef());
                break;
            case 3: // Recursive ARMA
                if (getSeriesCoef() == null) {
                    setSeriesCoef(new double[2*getSeriesDeg() + 2]);
                }
                ARMAKalmanRecursive(w,getSeriesLen(),getSeriesDeg(), 0.992, getSeriesCoef());
                break;
            case 4: // Online Convex Optimizer (Newton Online Step) ARIMA
                if (getSeriesCoef() == null) {
                    setSeriesCoef(new double[getSeriesDeg() + 1]);
                }
                ARIMAOnlineConvexOptimizer(w,getSeriesLen(),getSeriesDeg(), 0.3162, getSeriesCoef());
                break;
        }
    }

    // Classical Burg Maximum Entropy method
    // Burg's lattice-based method: solves the lattice filter equations using the
    // harmonic mean of forward and backward squared prediction errors.
    private void ARMaxEntropy(double[] series, int len, int deg, double[][] ar, double[] per, double[] pef, double[] h, double[] g){

        int j, n, nn, jj;
        double sn, sd;
        double t1, t2;

        for (j = 1; j <= len; j++){
            pef[j] = 0.0;
            per[j] = 0.0;
        }

        for(nn = 2; nn <= deg + 1; nn++){

            n = nn - 2;
            sn = 0.0;
            sd = 0.0;
            jj = len - n - 1;

            for (j = 1; j <= jj; j++){
                t1 = series[j+n] + pef[j];
                t2 = series[j-1] + per[j];

                sn -= 2.0 * t1 * t2;
                sd += (t1 * t1) + (t2 * t2);
            }

            g[nn] = sn / sd;
            t1 = g[nn];

            if(n != 0){
                for (j = 2; j < nn; j++){
                    h[j] = g[j] + (t1 * g[n - j + 3]);
                }
                for (j = 2; j < nn; j++){
                    g[j] = h[j];
                }
                jj--;
            }

            for (j = 1; j <= jj; j++){
                per[j] += (t1 * pef[j]) + (t1 * series[j + nn - 2]);
                pef[j] = pef[j + 1] + (t1 * per[j + 1]) + (t1 * series[j]);
            }

            for (j = 2; j <= nn; j++){
                ar[nn - 1][j - 1] = g[j];
            }
        }
    }

    // Least-squares approach using Yule Walker equation.
    // Minimizes the standard sum of squared forward-prediction errors.
    private void ARLestSquares(double[] series, int len, int deg, double[] coef){
        int i, j, k, hj, hi;
        double[][] mat = new double[deg][deg];

        for (i = 0; i < deg; i++){
            coef[i] = 0.0;
            for (j = 0; j < deg; j++){
                mat[i][j] = 0.0;
            }
        }

        for (i = deg - 1; i < len - 1; i++){
            hi = i + 1;
            for (j = 0; j < deg; j++){
                hj = i - j;
                coef[j] += (series[hi] * series[hj]);
                for (k = j; k < deg; k++){
                    mat[j][k] += (series[hj] * series[i - k]);
                }
            }
        }

        for (i = 0; i < deg; i++) {
            coef[i] /= (len - deg);
            for (j = i; j < deg; j++) {
                mat[i][j] /= (len - deg);
                mat[j][i] = mat[i][j];
            }
        }

        // solve the linear equations through Gaussian elimination
        LinearEqSolver(mat, coef, deg);
    }

    private void LinearEqSolver(double[][] mat, double[] vec, int n){
        int i, j, k, maxi;
        double vswap, max, h, pivot, q;
        double[] mswap;
        double[] hvec;

        for (i = 0; i < n - 1; i++){
            max = Math.abs(mat[i][i]);
            maxi = i;

            for (j = i + 1; j < n; j++){
                if((h = Math.abs(mat[j][i])) > max){
                    max = h;
                    maxi = j;
                }
            }

            if (maxi != i) {
                mswap     = mat[i];
                mat[i]    = mat[maxi];
                mat[maxi] = mswap;

                vswap     = vec[i];
                vec[i]    = vec[maxi];
                vec[maxi] = vswap;
            }

            hvec = mat[i];
            pivot = hvec[i];

            if (Math.abs(pivot) != 0.0) {
                for (j = i + 1; j < n; j++) {
                    q = -mat[j][i] / pivot;
                    mat[j][i] = 0.0;
                    for (k = i + 1; k < n; k++) {
                        mat[j][k] += q * hvec[k];
                    }
                    vec[j] += (q * vec[i]);
                }
            }
        }

        vec[n-1] /= mat[n-1][n-1];
        for (i = n - 2; i >= 0; i--) {
            hvec = mat[i];
            for (j = n - 1; j > i; j--) {
                vec[i] -= (hvec[j] * vec[j]);
            }
            vec[i] /= hvec[i];
        }
    }

    // Forgetting factor Kalman filter algorithms are more computationally intensive than gradient and
    // unnormalized gradient methods. However, they have better convergence properties.
    // Estimate recursively parameters of AR model using a general recursive prediction error algorithm
    // using a Kalman Filter with Forgetting Factor.
    private void ARKalmanRecursive(double[] time_series, int len, int deg, double lambda, double[] coef) {

        deg = deg + 1;

        SimpleMatrix series = new SimpleMatrix(len, 1);
        for (int sid = 0; sid < len; sid++) {
            series.set(sid, time_series[sid]);
        }

        SimpleMatrix yh = new SimpleMatrix(1, 1);
        yh.set(0.0);

        double epsi;

        SimpleMatrix th = new SimpleMatrix(deg, 1);
        for (int did = 0; did < th.getNumElements(); did++) {
            th.set(did, UtilEjml.EPS);
        }

        SimpleMatrix phi = new SimpleMatrix(deg + 1, 1);
        for (int did = 0; did < phi.getNumElements(); did++) {
            phi.set(did, 0.0);
        }

        SimpleMatrix phi_pre = new SimpleMatrix(deg, 1);
        for (int did = 0; did < phi_pre.getNumElements(); did++) {
            phi_pre.set(0.0);
        }

        SimpleMatrix p = SimpleMatrix.identity(deg).scale(10000);
        SimpleMatrix R1 = new SimpleMatrix(deg, deg);
        R1.zero();

        SimpleMatrix K = new SimpleMatrix(deg, deg);
        K.zero();

        SimpleMatrix gain = new SimpleMatrix(1,1);
        gain.zero();

        for(int data_iter = 0; data_iter < len; data_iter++){
            phi_pre = phi.extractMatrix(0, deg, 0, 1);
            yh = phi_pre.transpose().mult(th);
            epsi = series.get(data_iter) - yh.get(0);
            gain = (((phi_pre.transpose().mult(p)).mult(phi_pre)).plus(lambda));
            K = (p.mult(phi_pre)).divide(gain.get(0));
            p = (p.minus(K.mult(phi_pre.transpose()).mult(p))).divide(lambda).plus(R1);
            th = th.plus(K.scale(epsi));
            for(int vid = 0; vid < deg; vid++) {
                phi.set(vid + 1, 0, phi_pre.get(vid));
            }
            phi.set(0, series.get(data_iter));
        }
        for(int id_coef = 0; id_coef < deg; id_coef++) {
            coef[id_coef] = th.get(id_coef);
        }
    }

    // Estimate recursively parameters of ARMA model using a general recursive
    // prediction error algorithm using a Kalman Filter with Forgetting Factor.
    private void ARMAKalmanRecursive(double[] time_series, int len, int deg, double lambda, double[] coef){

        deg = deg + 1;

        int degARMA = 2*deg; // lags in both error and history

        SimpleMatrix series = new SimpleMatrix(len, 1);
        for (int sid = 0; sid < len; sid++) {
            series.set(sid, time_series[sid]);
        }

        SimpleMatrix seriesTil = new SimpleMatrix(2, deg + 1);
        SimpleMatrix ztil = new SimpleMatrix(2, 1);

        SimpleMatrix yh = new SimpleMatrix(1, 1);
        yh.set(0.0);

        double epsi;
        double series_eps;

        SimpleMatrix th = new SimpleMatrix(degARMA, 1);
        for (int did = 0; did < th.getNumElements(); did++) {
            th.set(did, UtilEjml.EPS);
        }

        SimpleMatrix thic = th.extractMatrix(deg, degARMA, 0, 1);

        SimpleMatrix phi = new SimpleMatrix(degARMA, 1);
        for (int did = 0; did < phi.getNumElements(); did++) {
            phi.set(did, 0.0);
        }

        SimpleMatrix phii = new SimpleMatrix(degARMA - 2, 1);

        SimpleMatrix psi = new SimpleMatrix(degARMA, 1);
        for (int did = 0; did < psi.getNumElements(); did++) {
            psi.set(did, 0.0);
        }

        SimpleMatrix psii = new SimpleMatrix(degARMA - 2, 1);

        SimpleMatrix psiia = psi.extractMatrix(0, deg, 0, 1);
        SimpleMatrix psiic = psi.extractMatrix(deg, degARMA, 0, 1);

        SimpleMatrix p = SimpleMatrix.identity(degARMA).scale(10000);
        SimpleMatrix R1 = new SimpleMatrix(degARMA, degARMA);
        R1.zero();

        SimpleMatrix K = new SimpleMatrix(degARMA, degARMA);
        K.zero();

        SimpleMatrix gain = new SimpleMatrix(1,1);
        gain.zero();

        Complex_F64[] polyCoeff = new Complex_F64[deg + 1];
        polyCoeff[0] = new Complex_F64(1.0, 0.0);

        Complex_F64[] stabCoeff = new Complex_F64[deg + 1];
        for (int cid = 0; cid < stabCoeff.length; cid++){
            stabCoeff[cid] = new Complex_F64(0.0,0.0);
        }
        SimpleMatrix stabCoeffMatrix = new SimpleMatrix(deg + 1, 1);

        for(int data_iter = 0; data_iter < len; data_iter++){

            psiia = psi.extractMatrix(0, deg, 0, 1);
            psiic = psi.extractMatrix(deg, degARMA, 0, 1);

            yh = phi.transpose().mult(th);
            epsi = series.get(data_iter) - yh.get(0);

            gain = (((psi.transpose().mult(p)).mult(psi)).plus(lambda));
            K = (p.mult(psi)).divide(gain.get(0));
            p = (p.minus(K.mult(psi.transpose()).mult(p))).divide(lambda).plus(R1);

            th = th.plus(K.scale(epsi));

            thic = th.extractMatrix(deg, degARMA, 0, 1);

            // stabilize the polynomial
            for(int did = 1; did < deg + 1; did++){
                polyCoeff[did] = new Complex_F64(thic.get(did-1), 0.0);
            }

            // numerical stabilization
            stabCoeff = polyStab(polyCoeff);

            for(int sid = deg; sid < degARMA; sid++) {
                th.set(sid, 0, stabCoeff[sid + 1 - deg].getReal());
            }

            series_eps = series.get(data_iter) - phi.transpose().mult(th).get(0);

            seriesTil.set(0, 0, series.get(data_iter));
            seriesTil.set(1, 0, series_eps);

            for (int zid = 1; zid < deg + 1; zid++) {
                seriesTil.set(0, zid, psiia.transpose().get(zid-1));
                seriesTil.set(1, zid, -psiic.transpose().get(zid-1));
            }

            for (int cid = 0; cid < stabCoeffMatrix.getNumElements(); cid++) {
                stabCoeffMatrix.set(cid, stabCoeff[cid].getReal());
            }
            ztil = seriesTil.mult(stabCoeffMatrix);

            for(int subid = 0; subid < phii.getNumElements()/2; subid++){
                phii.set(subid, phi.get(subid));
                psii.set(subid, psi.get(subid));
            }
            for(int supid = deg - 1; supid < phii.getNumElements(); supid++){
                phii.set(supid, phi.get(supid + 1));
                psii.set(supid, psi.get(supid + 1));
            }

            for(int subid = 1; subid < deg; subid++){
                phi.set(subid, phii.get(subid - 1));
                psi.set(subid, psii.get(subid - 1));
            }
            for(int supid = deg + 1; supid < degARMA; supid++){
                phi.set(supid, phii.get(supid - 2));
                psi.set(supid, psii.get(supid - 2));
            }

            phi.set(0, -series.get(data_iter));
            psi.set(0, -ztil.get(0));

            phi.set(deg, series_eps);
            psi.set(deg, ztil.get(1));

        }

        // collect coefficients
        for(int id_coef = 0; id_coef < degARMA; id_coef++) {
            coef[id_coef] = -th.get(id_coef);
        }

    }

    // Helper function to stabilize the Kalman Filter recursion in the ARMA Estimator
    private Complex_F64[] polyStab(Complex_F64[] a) {
        double[] coeff = new double[a.length];
        Complex_F64 stab_coeff[] = new Complex_F64[a.length];
        for(int rid = 0; rid < stab_coeff.length; rid++) {
            stab_coeff[rid] = new Complex_F64(0.0, 0.0);
        }

        for(int id = 0; id < a.length; id++){
            // coeffs are reversed order
            coeff[coeff.length-id-1] = a[id].getReal();
        }
        Complex_F64 r[] = findRoots(coeff);
        // reverse coefficients order
        Complex_F64 baser_r = r[0];
        for(int ri = 0; ri < r.length - 1; ri++){
            r[ri] = r[ ri + 1];
        }
        r[r.length - 1 ] = baser_r;

        double vs[] = new double[r.length];
        Complex_F64 r_conj[] = new Complex_F64[r.length];
        for(int rid = 0; rid < r_conj.length; rid++) {
            r_conj[rid] = new Complex_F64(0.0, 0.0);
        }

        for (int vid = 0; vid < vs.length; vid++) {
            vs[vid] = 0.5 * (java.lang.Math.signum(r[vid].getMagnitude() - 1.0) + 1.0);
        }

        for (int vid = 0; vid < vs.length; vid++) {
            ComplexMath_F64.conj(r[vid], r_conj[vid]);
        }

        // calculate history impact
        Complex_F64 r_hist[] = new Complex_F64[r.length];
        for(int rid = 0; rid < r_hist.length; rid++) {
            r_hist[rid] = new Complex_F64((1.0 - vs[rid])*r[rid].getReal(), (1.0 - vs[rid])*r[rid].getImaginary());
        }

        // calculate increment
        Complex_F64 r_inc[] = new Complex_F64[r.length];
        for(int rid = 0; rid < r_inc.length; rid++) {
            if(!r_conj[rid].isReal()) {
                r_inc[rid] = new Complex_F64(vs[rid], 0.0).divide(r_conj[rid]);
            }
            else{
                r_inc[rid] = new Complex_F64(vs[rid] / r_conj[rid].getReal(), 0.0);
            }
        }

        // update the root for stabilization
        for (int vid = 0; vid < vs.length; vid++) {
            r[vid].set(r_hist[vid].plus(r_inc[vid]));
        }
        // fit polynomial and multiply with gain
        for(int cid = 0; cid < stab_coeff.length; cid++) {
            stab_coeff[cid].set((polyFit(r)[cid]).getReal(), 0.0);
        }

        return stab_coeff;
    }

    // Given a set of polynomial coefficients, compute the roots of the polynomial.  Depending on
    // the polynomial being considered the roots may contain complex number.  When complex numbers are
    // present they will come in pairs of complex conjugates
    private Complex_F64[] findRoots(double... coefficients) {
        int N = coefficients.length-1;

        // Construct the companion matrix
        DMatrixRMaj c = new DMatrixRMaj(N,N);

        double a = coefficients[N];
        for(int i = 0; i < N; i++ ) {
            c.set(i,N-1,-coefficients[i]/a);
        }
        for(int i = 1; i < N; i++ ) {
            c.set(i,i-1,1);
        }

        // use generalized eigenvalue decomposition to find the roots
        EigenDecomposition_F64<DMatrixRMaj> evd =  DecompositionFactory_DDRM.eig(N,false);
        evd.decompose(c);

        Complex_F64[] roots = new Complex_F64[N];
        for(int i = 0; i < N; i++ ) {
            roots[i] = evd.getEigenvalue(i);
        }

        return roots;
    }

    // Create a polynomial with given roots.
    private Complex_F64[] polyFit(Complex_F64[] roots) {

        int n = roots.length;
        Complex_F64[] poly = new Complex_F64[n+1];
        Complex_F64[] update = new Complex_F64[n];

        for (int id = 0; id < poly.length; id++){
            poly[id] = new Complex_F64(0.0, 0.0);
        }
        poly[0].set(new Complex_F64(1.0, 0.0));

        for (int id = 0; id < update.length; id++){
            update[id] = new Complex_F64(0.0, 0.0);
        }

        for(int cid = 0; cid < n; cid++){
            // expand recursive formulation
            for(int subid = 0; subid <= cid; subid++){
                update[subid] = roots[cid].times(poly[subid]);
            }
            for(int supid = 0; supid <= cid; supid++) {
                poly[supid + 1] = poly[supid + 1].minus(update[supid]);
            }
        }
        // result should be real only if the roots are complex conjugates
        int conj_pairs = 0;
        for (Complex_F64 rid:roots){
            if(rid.getImaginary()>0){
                conj_pairs++;
            }
        }
        if(conj_pairs!=0) {
            for (Complex_F64 cid: poly) {
                cid.set(cid.getReal(), 0.0);
            }
        }
        return poly;
    }

    // Online ARIMA estimation using game-theoretic framework ,
    // where an online player sequentially commits to a decision and
    // then suffers from a loss which may be unknown to the decision
    // maker ahead of time.
    // In the online setting of ARIMA, we assume coefficient vectors (α, β) are fixed by the adversary.
    // Follow the idea of improper learning principle to design a
    // solution where the prediction does not come directly from
    // the original ARIMA model, but from a modified ARIMA
    // model (without the explicit noise terms) that approximates
    // the original model using an online convex optimization solver (Online Newton Step).
    private void ARIMAOnlineConvexOptimizer(double[] time_series, int len, int deg, double lambda, double[] coef){

        deg = deg + 1;
        double learningRate = 1.75;

        SimpleMatrix coeffUpdate = new SimpleMatrix(1, deg);
        for(int id = 0; id < coeffUpdate.getNumElements(); id++){
            coeffUpdate.set(id, Math.random());
        }

        SimpleMatrix aTrans = SimpleMatrix.identity(deg).scale(lambda);
        SimpleMatrix aTransUpdate = new SimpleMatrix(deg, deg);
        aTransUpdate.zero();

        SimpleMatrix dataWin = new SimpleMatrix(1, deg);

        double diff;
        double ratioDiv;
        SimpleMatrix grad;

        for(int data_iter = deg; data_iter < len; data_iter++){

            for(int did = data_iter - deg, wid = 0; did < data_iter; wid++, did++){
                dataWin.set(wid, time_series[did]);
            }

            diff = coeffUpdate.mult(dataWin.transpose()).minus(time_series[data_iter]).get(0);
            grad = dataWin.scale(2.0*diff);

            ratioDiv = grad.mult(aTrans).mult(grad.transpose()).get(0) + 1.0;

            aTransUpdate = (aTrans.mult(grad.transpose().mult(grad)).mult(aTrans)).divide(ratioDiv);

            aTrans = aTrans.minus(aTransUpdate);

            coeffUpdate = coeffUpdate.minus(grad.mult(aTrans).scale(learningRate));
        }
        for(int cid = 0; cid < deg; cid++){
            coef[cid] = coeffUpdate.get(cid);
        }
    }
}
