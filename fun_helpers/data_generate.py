import numpy as np
from skfda.misc.covariances import Exponential
from skfda.datasets import make_gaussian_process
from scipy.stats import norm
from scipy.stats import beta

##################################################################################
############## Here are some math functions used in data generation ##############
##################################################################################

def sigmoid(t):
	return 2 / (1 + np.exp(-t)) - 1  

def tanh_function(t):
	return np.tanh(t) 

def new_pdf_1(times, t_long, w_n = 0.6, w_b = 0.4, mu = 0, sigma = 1, alpha = 2, beta_val = 3):
    pdf_norm_1 = norm.pdf(t_long*times, mu, sigma)
    pdf_beta_1 = beta.pdf(t_long*times, alpha, beta_val)
    X1 = w_n*pdf_norm_1 + w_b*pdf_beta_1
    return X1

def new_pdf_2(times, t_long, w_n = 0.3, w_b = 0.3, mu = 0.5, sigma = 0.5, alpha = 3, beta_val = 4):
    pdf_norm_1 = norm.pdf(t_long*times, mu, sigma)
    pdf_beta_1 = beta.pdf(t_long*times, alpha, beta_val)
    X2 = w_n*pdf_norm_1 + w_b*pdf_beta_1
    return X2

def gap_x(gap_len, max_x):
    x_zero = np.zeros(gap_len)
    gap_time = np.linspace(0,max_x,100-gap_len)
    return np.concatenate([x_zero, gap_time])

def new_pdf_3(times, pow=4,gap_len=45,max_x=0.98):
    X1 = times ** pow - gap_x(gap_len, max_x) 
    return X1

def new_pdf_4(times,pow=5, gap_len=55, max_x=1.02):
    X2 = times ** pow - gap_x(gap_len, max_x) 
    return X2

##################################################################################
###################################### End #######################################
##################################################################################


class FunctionalGenerator(object):
	"""
    Generate synthetic functional data for binary classification tasks.
    
    Each sample is a multi-dimensional function X: [0,1] → R^d, generated as a sum of smooth signals and Gaussian 
	process noise. The first n samples are labeled 0 and follow one class of underlying patterns, and the remaining 
	n samples are labeled 1 with a different set of patterns.

    Parameters
    ----------
    npoints : int, default=100
        Number of evaluation points per functional sample (i.e., time grid size).
		
    w_noise : float, default=0.5
        Scaling factor for the additive Gaussian process noise.
		
    seed : int or None
        Random seed for reproducibility.
    """

	
	def __init__(self, npoints = 100, w_noise = 0.5, seed = None):
		self.npoints = npoints
		self.w_noise = w_noise
		if seed:
			np.random.seed(seed)


	def get_data(self, n, d):
		"""
        Generate 2n functional data samples with d-dimensional features, half from each class.
        
        Class 0 and class 1 functions differ in their underlying signal structure. Each feature dimension corresponds 
        to a specific time-dependent function. Additive Gaussian process noise is applied across all dimensions.

        Feature definitions (per time t ∈ [0,1]):
        
        For label Y = 0:
            X_0(t) = exp(cos(2πt)) / 2
            X_1(t) = 1.6 * t^{1/3}
            X_2(t) = log(0.5 + cos(πt^4 / 2)) + 1
            X_3(t) = exp(sin(2πt)) / 2
            X_4(t) = 0.6 * N(0,1)(t) + 0.4 * Beta(2,3)(t) + 0.5
            X_5(t) = t^4 - g(t; 0.55) + 1
            X_6(t) = -0.15 * t - 0.2 * t^2 + 0.95
            X_7(t) = sigmoid(20t - 10)/3 + 0.5

        For label Y = 1:
            X_0(t) = exp(cos(2πt^{1.05})) / 2
            X_1(t) = sqrt(3t)
            X_2(t) = 0.9 * log(0.5 + cos(πt^3 / 2)) + 1
            X_3(t) = exp(sin(2πt^{1.05})) / 2
            X_4(t) = 0.3 * N(0.5, 0.5)(t) + 0.3 * Beta(3,4)(t) + 0.5
            X_5(t) = t^5 - g(t; 0.45) + 1
            X_6(t) = -0.5 * t + 0.2 * t^2 + 0.95
            X_7(t) = tanh(12t - 6.3)/3 + 0.5

        Parameters
        ----------
        n : int
            Number of samples per class (total 2n).
			
        d : int
            Feature dimension (d ≤ 8 supported by default design).

        Returns
        -------
        X : ndarray of shape (2n, npoints, d)
            Functional samples for each instance over time.
			
        y : ndarray of shape (2n,)
            Corresponding binary class labels (0 or 1).
        """		
		
		X = np.zeros((2*n, self.npoints,d))
		y = np.zeros((2*n))
		time = np.linspace(0, 1, num=self.npoints)

        # Add Gaussian process noise to feature channels
		for i in range(2*n):
			gp = make_gaussian_process(n_features=self.npoints, n_samples=d, cov=Exponential())
			noise = gp.data_matrix.T * self.w_noise
			X[i, :, :] = noise
               
        # Class 0 generation
		if d > 0: X[:n, :, 0] += np.exp(np.sin(2 * np.pi * (time)))/2  
		if d > 1: X[:n, :, 1] += 1.6 * (time ** (1/3))  
		if d > 2: X[:n, :, 2] += np.log(0.5 + np.cos(np.pi * np.power(time, 4) / 2)) + 1 
		if d > 3: X[:n, :, 3] += np.exp(np.cos(2 * np.pi * (time)))/2  
		if d > 4: X[:n, :, 4] += new_pdf_1(time, t_long=5) + 0.5 
		if d > 5: X[:n, :, 5] += new_pdf_3(time) + 1 
		if d > 6: X[:n, :, 6] += -0.15 * time - 0.2 * time ** 2 + 0.95  
		if d > 7: X[:n, :, 7] += sigmoid(20 * time - 10)/3 + 0.5   
		y[:n] = 0
          
        # Class 1 generation
		if d > 0: X[n:, :, 0] += np.exp(np.sin(2 * np.pi * (np.power(time, 1.05))))/2 
		if d > 1: X[n:, :, 1] += np.sqrt(3 * time)  
		if d > 2: X[n:, :, 2] += 0.9 * np.log(0.5 + np.cos(np.pi * np.power(time, 3) / 2)) + 1
		if d > 3: X[n:, :, 3] += np.exp(np.cos(2 * np.pi * (np.power(time, 1.05))))/2  
		if d > 4: X[n:, :, 4] += new_pdf_2(time, t_long=5) + 0.5 
		if d > 5: X[n:, :, 5] += new_pdf_4(time) + 1  
		if d > 6: X[n:, :, 6] += -0.5 * time + 0.2 * time ** 2 + 0.95 
		if d > 7: X[n:, :, 7] += 0.95*tanh_function(12 * time - 6.3)/3 + 0.5 
		y[n:] = 1

		return X, y



def time_uneven(length=100, intv=0.01, sigma=0.3):
    """
    Generate a normalized, unevenly spaced time grid over [0, 1].

    Each interval between time points is randomly drawn from a normal distribution,
    ensuring a non-uniform temporal resolution.

    Parameters
    ----------
    length : int, default=100
        Number of time points to generate.

    intv : float, default=0.01
        Base interval added to each time increment (minimum spacing).

    sigma : float, default=0.3
        Standard deviation for randomness in interval lengths.

    Returns
    -------
    new_time : ndarray of shape (length,)
        Non-uniformly spaced and normalized time grid from 0 to 1.
    """

    increments = intv + np.abs(np.random.normal(loc=1-intv, scale=sigma, size=length-1))
    cumulative_sum = np.cumsum(increments)
    time_series = np.insert(cumulative_sum, 0, 0)
    new_time = (time_series - time_series[0]) / (time_series[-1] - time_series[0])
    return new_time



class FunctionalGenerator_Unevenly_Sampling(object):
    """
    Generate synthetic functional data with *unevenly sampled* time grids for binary classification.

    Each sample is a 2-dimensional function X: [0,1] → R^d with d=2 default features and an additional
    timestamp channel. Data is corrupted by Gaussian process noise. Two classes (Y=0 and Y=1) are 
    generated with different underlying signal structures.

    Parameters
    ----------
    npoints : int, default=100
        Number of sample points per functional observation.

    w_noise : float, default=0.5
        Scaling factor for the additive Gaussian process noise.

    seed : int or None
        Random seed for reproducibility.
    """
    
    def __init__(self, npoints=100, w_noise=0.5, seed=None):
        self.npoints = npoints
        self.w_noise = w_noise
        if seed:
            np.random.seed(seed)

    def get_data(self, n, d=2, intv=0.01, sigma=0.3):
        """
        Generate 2n functional data samples with d noisy feature dimensions + 1 time dimension.
        
        Functional data is sampled on uneven time grids unique to each sample, simulating irregular
        sampling in practical scenarios. Two different signal-generating processes are used for
        class 0 and class 1.

        Feature definitions (for each t ∈ [0,1]):
        
        For y = 0:
            X_0(t) = exp(sin(2πt)) / 2 + noise
            X_1(t) = 1.6 * t^{1/3} + noise
            X_2(t) = time (non-uniform)

        For y = 1:
            X_0(t) = exp(sin(2πt^{1.05})) / 2 + noise
            X_1(t) = sqrt(3t) + noise
            X_2(t) = time (non-uniform)

        Parameters
        ----------
        n : int
            Number of samples per class (total 2n).

        d : int, default=2
            Number of noisy feature dimensions (excluding time dimension).
            
        intv : float, default=0.01
            Base interval used to generate uneven time spacing.

        sigma : float, default=0.3
            Standard deviation for the random fluctuation of time intervals.

        Returns
        -------
        X : ndarray of shape (2n, npoints, d+1)
            Functional observations. The last channel (index d) stores the time grid for each sample.

        y : ndarray of shape (2n,)
            Corresponding binary class labels (0 or 1).
        """
        
        pow = 1.05

        X = np.zeros((2 * n, self.npoints, d + 1))
        y = np.zeros((2 * n))

        # Add Gaussian process noise to feature channels
        for i in range(2 * n):
            gp = make_gaussian_process(n_features=self.npoints, n_samples=d, cov=Exponential())
            noise = gp.data_matrix.T * self.w_noise 
            X[i, :, :2] = noise

        # Class 0 generation
        for i in range(n):
            time1 = time_uneven(length=self.npoints, intv=intv, sigma=sigma)
            X[i, :, 0] += np.exp(np.sin(2 * np.pi * (time1))) / 2
            X[i, :, 1] += 1.6 * (time1 ** (1 / 3))
            X[i, :, 2] += time1
            y[i] = 0

        # Class 1 generation
        for i in range(n):
            time2 = time_uneven(length=self.npoints, intv=intv, sigma=sigma)
            X[n + i, :, 0] += np.exp(np.sin(2 * np.pi * ((time2 ** pow)))) / 2
            X[n + i, :, 1] += np.sqrt(3 * time2)
            X[n + i, :, 2] += time2
            y[n + i] = 1

        return X, y


class ScalarGenerator(object):
    """
    Generate simulated scalar features for binary classification.

    This class simulates `q`-dimensional scalar features for `2 * n` samples (balanced classes).
    Each feature dimension is generated from a different pair of distributions for class 0 and class 1.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed):
        self.seed = seed

    def get_data(self, n, q=1):
        """
        Generate scalar features with distinct distributions for two classes.

        Each sample is a vector of length `q`, where each feature dimension follows
        a pair of different distributions depending on the class label (y=0 or y=1).
        The first `n` samples belong to class 0 (y=0), the next `n` samples to class 1 (y=1).

        Supported distributions (D_i) per feature dimension:
        -----------------------------------------------------
        - z_1: U(1, 2)         vs. U(0.75, 1.75)
        - z_2: N(0, 1)         vs. N(0.5, 1)
        - z_3: Exp(λ=2)        vs. Exp(λ=1)       (λ = 1 / scale)
        - z_4: χ²(df=0.1)      vs. χ²(df=0.2)
        - z_5: LogN(0, 1)      vs. LogN(0.25, 1)
        - z_6: Gamma(2, 2)     vs. Gamma(3, 2)
        - z_7: Beta(2, 3)      vs. Beta(3, 2)
        - z_8: Bernoulli(0.55) vs. Bernoulli(0.45)

        Parameters
        ----------
        n : int
            Number of samples per class.

        q : int, default=1
            Dimension of scalar features to generate (max = 8).

        Returns
        -------
        z : ndarray of shape (2 * n, n_scalar)
            Matrix of simulated scalar features, with the first n rows corresponding to class 0,
            and the next n rows to class 1.
        """
        np.random.seed(self.seed)
        z = np.zeros((2 * n, q))

        # Generate features for class 0 (first ntrain rows)
        for i in range(n):
            if q > 0: z[i, 0] = np.random.uniform(1, 2)
            if q > 1: z[i, 1] = np.random.normal(0, 1)
            if q > 2: z[i, 2] = np.random.exponential(0.5)
            if q > 3: z[i, 3] = np.random.chisquare(0.1)
            if q > 4: z[i, 4] = np.random.lognormal(0, 1)
            if q > 5: z[i, 5] = np.random.gamma(2, 2)
            if q > 6: z[i, 6] = np.random.beta(2, 3)
            if q > 7: z[i, 7] = np.random.binomial(n=1, p=0.55)

        # Generate features for class 1 (next ntrain rows)
        for j in range(n):
            if q > 0: z[n + j, 0] = np.random.uniform(0.75, 1.75)
            if q > 1: z[n + j, 1] = np.random.normal(0.5, 1)
            if q > 2: z[n + j, 2] = np.random.exponential(1)
            if q > 3: z[n + j, 3] = np.random.chisquare(0.2)
            if q > 4: z[n + j, 4] = np.random.lognormal(0.25, 1)
            if q > 5: z[n + j, 5] = np.random.gamma(3, 2)
            if q > 6: z[n + j, 6] = np.random.beta(3, 2)
            if q > 7: z[n + j, 7] = np.random.binomial(n=1, p=0.45)

        return z
      
