import numpy as np
import iisignature as isig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from fun_helpers.tools import get_sigX, get_sigX_timeMis, data_normal


class PSLR(object):
    """Path-Signature based Logistic Regression Model

    A logistic regression model that combines path-signature features from time-augmented
    paths with scalar covariates, regularized using L1 penalty.

    Parameters
    ----------
    p : int
        Truncation order of the path signature. Higher values capture more path information.

    pen_lambda : float, optional (default=1)
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization (L1).
    
    Attributes
    ----------
    reg : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model with L1 regularization.
    """

    def __init__(self, p, pen_lambda = 1):
        self.p = p
        self.reg = LogisticRegression(solver='liblinear', penalty='l1', C = pen_lambda)


    def fit(self, X, y, z = None):
        """Fit the logistic regression model using path-signature and scalar covariates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, d)
            3D array of time-augmented paths. Each path is piecewise linear in R^d
            with `n_points` time steps.

        y : array-like of shape (n_samples,)
            Binary class labels (0 or 1) for each sample.

        z : array-like of shape (n_samples, q)
            2D array of scalar covariates for each sample.

        Returns
        -------
        reg : sklearn.linear_model.LogisticRegression
            The fitted logistic regression model.
        """
        sigX = get_sigX(X, self.p)
        if z is None :
            Tilde_sigX = sigX
        else:
            Tilde_sigX = np.concatenate((z,sigX), axis = 1)
        Tilde_sigX = data_normal(Tilde_sigX)
        self.reg.fit(Tilde_sigX, y)
        return self.reg
    

    def predict(self, X, z = None):
        """Predict class labels using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, d)
            3D array of time-augmented paths for prediction.

        z : array-like of shape (n_samples, q)
            2D array of scalar covariates for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1) for each sample.
        """
        sigX = get_sigX(X, self.p)
        if z is None :
            Tilde_sigX = sigX
        else:
            Tilde_sigX = np.concatenate((z,sigX), axis = 1)
        Tilde_sigX = data_normal(Tilde_sigX)
        y_pred = self.reg.predict(Tilde_sigX)
        return y_pred
    
    
    def get_loss(self, X, y, z = None):
        """Compute average negative log-likelihood (log loss) on the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, d)
            3D array of time-augmented input paths.

        y : array-like of shape (n_samples,)
            Binary class labels (0 or 1) for each sample.

        z : array-like of shape (n_samples, q)
            2D array of scalar covariates.

        Returns
        -------
        loss : float
            Mean negative log-likelihood (log loss) over all samples.
        """
        sigX = get_sigX(X,self.p)
        if z is None :
            Tilde_sigX = sigX
        else:
            Tilde_sigX = np.concatenate((z,sigX), axis = 1)
        Tilde_sigX = data_normal(Tilde_sigX)
        self.reg.fit(Tilde_sigX,y)
        coef = self.reg.coef_
        inner = coef * Tilde_sigX
        inn = sum(inner.T)
        Logloss = y * inn - np.log(1+np.exp(inn))
        return - sum(Logloss)/ np.size(y)
    


class PSLR_Order(object):
    """Model selection class for determining optimal signature order in PSLR.

    This class selects the best truncation order `p` for the Path-Signature based Logistic Regression (PSLR)
    model based on penalized empirical loss and visual slope heuristic.

    Parameters
    ----------
    rho : float, optional (default=0.4)
        Regularization rate in the penalty term. Controls how the penalty scales with sample size.

    pen_lambda : float, optional (default=1)
        L1 regularization strength used in the inner PSLR models.

    Notes
    -----
    Part of this implementation is adapted from:
    https://github.com/afermanian/signature-regression
    """
    def __init__(self, rho = 0.4, pen_lambda = 1):

        self.pen_lambda = pen_lambda
        self.rho = rho


    def Loss(self, Pmax, X, y, z= None):
        """Compute empirical loss for all signature orders from 0 to Pmax.

        Parameters
        ----------
        Pmax : int
            Maximum signature order to consider.

        X : array-like of shape (n_samples, n_points, d)
            Time-augmented paths.

        y : array-like of shape (n_samples,)
            Binary class labels.

        z : array-like of shape (n_samples, q)
            Scalar covariates.

        Returns
        -------
        Loss : ndarray of shape (Pmax + 1,)
            Logistic losses for signature orders 0 through Pmax.
        """
        Loss = np.zeros(Pmax + 1)

        for i in range(Pmax + 1):
            Model = PSLR(i, pen_lambda = self.pen_lambda)
            Model.fit(X, y, z)
            Loss[i] = Model.get_loss(X, y, z)

        return Loss


    def get_pen(self, n, q, d, p, Cpen):
        """Compute the penalty term for a given signature order.

        Parameters
        ----------
        n : int
            Number of training samples.

        q : int
            Dimension of scalar covariates.

        d : int
            Dimension of the path (within time augmentation).

        p : int
            Signature truncation order.

        Cpen : float
            Penalty scaling constant.

        Returns
        -------
        penalty : float
            Computed penalty value for the given signature order.
        """
        if p == 0:
            size_sig = 1
        else:
            size_sig = isig.siglength(d, p) + 1

        return Cpen * n ** (-self.rho) * np.sqrt(size_sig * np.exp(q))
    

    def select_order_p(self, P_max, X, y, z = None, CpenMax = 1 * 10 ** 0):
        """Select the optimal signature order `p` using penalized loss and slope heuristic.

        This method computes the penalized empirical loss across signature orders `0` to `P_max`,
        plots the model complexity path with respect to penalty values, and prompts the user
        to manually select the optimal order based on slope heuristic or visual cues.

        Parameters
        ----------
        P_max : int
            Maximum signature order to search over.

        X : array-like of shape (n_samples, n_points, d)
            Time-augmented paths.

        y : array-like of shape (n_samples,)
            Binary class labels.

        z : array-like of shape (n_samples, q)
            Scalar covariates.

        CpenMax : float, optional (default=1.0)
            Maximum penalty coefficient to consider in the search grid.

        Returns
        -------
        hatp : int
            Signature order `p` manually selected by the user,
            based on visual inspection of the plot and slope heuristic.
        """
        num_train, _, d = X.shape
        if z is None:
            q = 0
            Loss = self.Loss(P_max, X, y)
        else:
            _, q = z.shape
            Loss = self.Loss(P_max, X, y, z)


        Cpen_values = np.linspace(0, CpenMax, 1000)
        hatp = np.zeros(len(Cpen_values))

        for i in range(len(Cpen_values)):
            pen = np.zeros(P_max + 1)
            for j in range(P_max + 1):
                pen[j] = self.get_pen(num_train, q, d, j, Cpen_values[i])
            hatp[i] = np.argmin(Loss + pen)
                

        palette = sns.color_palette('colorblind')

        fig, ax = plt.subplots(figsize=(4, 2.5))
        jump = 1
        for i in range(P_max + 1):
            if i in hatp:
                xmin = Cpen_values[hatp == i][0]
                xmax = Cpen_values[hatp == i][-1]
                ax.hlines(i, xmin, xmax, colors='b')
                if i != 0:
                    ax.vlines(xmax, i, i - jump, linestyles='dashed', colors=palette[0])
                jump = 1
            else:
                jump += 1
        ax.set_xlabel('$C_{pen}$')
        ax.set_ylabel(r'$\hat{p}$', rotation=0, labelpad = 5)

        plt.title("PSLR Order Selection")
        plt.grid()
        plt.show()

        hatp = float(input("Enter the selected hatp by slope heuristic result:"))

        return int(hatp)
