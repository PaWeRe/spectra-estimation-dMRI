
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from cvxopt import matrix, solvers

# make a class that contains signal values and associated b_values
# P edit: Add snr (based on voxel count)
# P edit 2: Add patient identifier, as class is created for 63 patients
class signal_data:
    def __init__(self, signal_values, b_values):
        self.signal_values = signal_values
        self.b_values = b_values
        # self.v_count = v_count
        # self.patient_id = patient_id
    def plot(self, save_filename=None, title=None):
        plt.plot(self.b_values, self.signal_values, linestyle='None', marker='.')
        plt.xlabel('B Value')
        plt.ylabel('Signal')
        if title is not None:
            plt.title(title)
        #plt.show()
        if save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()
    #Plot 3rd dimension (snr/p_count as well)
    def plot3d(self, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.b_values, self.signal_values, self.v_count, marker='.')
        ax.set_xlabel('B Value')
        ax.set_ylabel('Signal')
        ax.set_zlabel('V Count')
        if title is not None:
            ax.set_title(title)
        plt.show()

# package up fractions and diffusivities
class d_spectrum:
    def __init__(self, fractions, diffusivities):
        self.fractions = fractions
        self.diffusivities = diffusivities
    def plot(self, title=None):
        # plt.plot(diffusivities, fractions, 'o')
        plt.vlines(self.diffusivities, np.zeros(len(self.fractions)), self.fractions)
        plt.xlabel(r'Diffusivity Value ($\mu$m$^2$/ ms.)')
        plt.ylabel('Relative Fraction')
        plt.xlim(left=0)
        if title is not None:
            plt.title(title)
        plt.show()


# a collection of diffusivity spectra, probably
# result of Gibbs' sampling
class d_spectra_sample:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.sample = [] # a list of samples
    def plot(self, save_filename=None, title=None, start=0, end=-1, skip=False, ax=None):
        if not ax:
            fig, ax = plt.subplots()
            ax.set_xlabel(r'Diffusivity Value ($\mu$m$^2$/ ms.)')
            ax.set_ylabel('Relative Fraction')
            tick_labels = []
            for number in self.diffusivities:
                tick_labels.append(str(number))
            if skip:
                for i in range(0, len(self.diffusivities)):
                    if i%2 == 1:
                        tick_labels[i] = ''
            print(tick_labels)
            ax.set_xticklabels(tick_labels, rotation=45)
        if title is not None:
            ax.set_title(title)
        sample_arr = np.asarray(self.sample)[start:end]
        ax.boxplot(sample_arr, showfliers=False, manage_ticks=False, showmeans=True, meanline=True)
        # store = ax.boxplot(sample_arr, showfliers=True, manage_ticks=False, showmeans=True, meanline=True, whis=10)
        # sum_means = np.sum([store['means'][i]._y[0] for i in range(10)])
        # print(sum_means)
        if save_filename is not None:
            plt.savefig(save_filename)
        
    def diagnostic_plot(self):
        plt.plot(np.asarray(self.sample))
        plt.show()
    def normalize(self):
        for i in range(0, len(self.sample)):
            sum = np.sum(self.sample[i])
            self.sample[i] = self.sample[i] / sum

# A multi compartment signal generator
#
# predict a single or multiple signals corresponding to the specified b value(s),
# fractions are the mixing coefficients of the corresponding diffusivities
# they are supposed to sum to one
def predict_signal(s_0, b_value, diffusivities, fractions):
    signal = 0
    for i in range(len(diffusivities)):
        signal += s_0 * fractions[i] * np.exp(- b_value * diffusivities[i])
    return(signal)
# make a vector of signals that correspond to a vector of b_values
# with optional noise added
# (with noise, sigs can be negative)
def generate_signals(s_0, b_values, d_spectrum, sigma=0.0):
    predictions = predict_signal(s_0, b_values, d_spectrum.diffusivities, d_spectrum.fractions)
    if sigma > 0.0:
        predictions += npr.normal(0, sigma, len(predictions))
    return signal_data(predictions, b_values)


###################################################################
# Calculate the parameters of the multivariate normal that characterizes
# the posterior distribution on R, the 'concentrations'
# with optional prior inverse covariance
# and optional L2 regularizer, which overides inverse_prior_covariance
# calculating the mean entails solving a linear equation in Sigma_inverse
# which may not be full rank.  Prints rank, if deficient can be fixed
# by using L1_lambda > 0.0

def calculate_MVN_posterior_params(signal_data, recon_diffusivities, sigma, inverse_prior_covariance=None, L1_lambda=0.0, L2_lambda=0.0):
    M_count = signal_data.signal_values.shape[0]
    N = recon_diffusivities.shape[0]
    u_vec_tuple = ()
    for i in range(M_count):
        u_vec_tuple += (np.exp((- signal_data.b_values[i] * recon_diffusivities)),)
    U_vecs = np.vstack(u_vec_tuple)
    U_outer_prods = np.zeros((N,N))
    for i in range(M_count):
        U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
    Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
    if L2_lambda > 0.0:
        inverse_prior_covariance = L2_lambda * np.eye(N)
    if inverse_prior_covariance is not None:
        Sigma_inverse += inverse_prior_covariance
    print('Size: {}, Rank: {}'.format(Sigma_inverse.shape[0], np.linalg.matrix_rank(Sigma_inverse)))
    Sigma = np.linalg.inv(Sigma_inverse)
    weighted_U_vecs = np.zeros(N)
    for i in range(M_count):
        weighted_U_vecs += signal_data.signal_values[i] * U_vecs[i]
    One_vec = np.ones(N)
    Sigma_inverse_M = (1.0 / (sigma * sigma) * weighted_U_vecs) - L1_lambda *  One_vec
    # avoid inverse
    M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
    return(M, Sigma_inverse, Sigma_inverse_M)
    
# find mode of truncated MVN by convex optimization
# def calculate_trunc_MVN_mode(M, Sigma, Sigma_inverse, Sigma_inverse_M):
def calculate_trunc_MVN_mode(M, Sigma_inverse, Sigma_inverse_M):
    N = Sigma_inverse.shape[0]
    P = matrix(Sigma_inverse)
    Q = matrix(- Sigma_inverse_M)
    G = matrix(- np.identity(N), tc='d')
    H = matrix(np.zeros(N))
    sol = solvers.qp(P, Q, G, H)
    mode = sol['x']
    return mode
    #  M, Sigma_inverse, and mode are established

# given signals etc, calculate the MVN parameters, then solve for the mode
def trunc_MVN_mode(signal_data, recon_diffusivities, sigma, inverse_prior_covariance=None, L1_lambda=0.0, L2_lambda=0.0):
    M, Sigma_inverse, Sigma_inverse_M = calculate_MVN_posterior_params(signal_data, recon_diffusivities, sigma, 
                                                                                inverse_prior_covariance, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
    mode =  calculate_trunc_MVN_mode(M, Sigma_inverse, Sigma_inverse_M)
    return d_spectrum(np.asarray(mode), recon_diffusivities)


#############################################################################
## this one had problems, so coded custom one (next)
## sample from specified univariate normal dist truncated to non zero values
# def sample_normal_non_neg(mu, sigma):
#     z = mu / sigma
#     # if z < -10.0:
#     if z < -5.0:
#         value = 0.0
#     else:
#         X = stats.truncnorm(
#             - mu / sigma, np.inf, loc=mu, scale=sigma)
#         value = X.rvs(1)[0]
#     # print('mu: {}, sigma: {}, value: {}'.format(mu, sigma, value))
#     return value
#############################################################################

# sample from a univariate normal distribution
# that is truncated to be non-negative
def sample_normal_non_neg_new(mu, sigma):
    if mu >= 0:
        # standard rejection sampler
        while True:
            u = npr.normal(mu, sigma)
            if u > 0.0:
                return(u)
    else:
        # mu is negative, use 'robert sampler', 
        # a sampler for a standard normal truncated on the left
        # at sigma_minus, which should be non negative
        # follows method in:
        #     'Simulation of Truncated Normal Variables'
        #     Cristian P. Robert, ArXiv 2009
        sigma_minus = - mu / sigma
        alpha_star = (sigma_minus + np.sqrt(sigma_minus**2 + 4)) / 2.0
        while True:
            x = npr.exponential(1.0 / alpha_star)  # arugment is 1 / lambda
            z = sigma_minus + x
            eta =  np.exp(- ((z - alpha_star)**2) / 2.0)
            u = npr.uniform()
            if u <= eta:
                # z is result of robert sampler, proposition 2.3 in paper
                # transform result back for more general non standard normal
                return mu + sigma * z



def make_Gibbs_sampler(signal_data,  diffusivities, sigma, inverse_prior_covariance=None, L1_lambda=0.0, L2_lambda=0.0):
    signals = signal_data.signal_values
    b_values = signal_data.b_values
    M, Sigma_inverse, weighted_U_vecs = calculate_MVN_posterior_params(signal_data, diffusivities, sigma, 
                                                                       inverse_prior_covariance, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
    # initialize R to mode of the truncated MVN
    R = np.array(calculate_trunc_MVN_mode(M, Sigma_inverse, weighted_U_vecs)).T[0]
    N = Sigma_inverse.shape[0]
    ###################################################
    print('\n\nSetting Up Gibbs Sampler')
    sigma_i                = np.empty(N, dtype=object)
    Sigma_inverse_quotient = np.empty(N, dtype=object)
    M_slash_i              = np.empty(N, dtype=object)
    for i in range(N):
        Sigma_inverse_ii = Sigma_inverse[i][i]
        sigma_i[i] = np.sqrt(1.0 / Sigma_inverse_ii)
        Sigma_inverse_i_slash_i =  np.delete(Sigma_inverse[i], i, axis=0) 
        Sigma_inverse_quotient[i] = Sigma_inverse_i_slash_i / Sigma_inverse_ii
        M_slash_i[i] = np.delete(M, i, 0)
    def Gibbs_sampler(iterations, the_sample=None):
        if the_sample is None:
            the_sample = d_spectra_sample(diffusivities)
        count = 0
        for j in range(iterations):
            if (count % 100) == 0:
                print('.', end='')
            count += 1
            for i in range(N):
            # for i in range(1):
                # next 3 are rowvecs...
                R_slash_i = np.delete(R, i, 0)
                dot_prod = np.dot(Sigma_inverse_quotient[i], (M_slash_i[i] - R_slash_i))
                # Patrick wonders if dot_prod needs index by i
                mu_i = M[i] + dot_prod
                # value = sample_normal_non_neg(mu_i, sigma_i[i])
                value = sample_normal_non_neg_new(mu_i, sigma_i[i])                
                R[i] = value
            # sample_collection.append(np.copy(R))
            the_sample.sample.append(np.copy(R))
        return the_sample
    return(Gibbs_sampler)



###########################################################################
# Gaussian process stuff
# Kernel for GP prior
def nep_kernel(i, j, k_sigma, scale):
    return np.square( k_sigma) * np.exp(-np.square((i - j) / scale) / 2.0 )
    

# covariance for GP prior
def make_prior_cov(k_sigma, scale, dims):
    def inner_nep_kernel(i,j):
        return nep_kernel(i,j, k_sigma, scale)
    return np.fromfunction(inner_nep_kernel, (dims, dims))



############################################################################
def predict_signals_from_diffusivity_sample(d_specta_sample, b_values):
    diffusivities = d_spectra_sample.diffusivities
    for i in range( len(d_spectra_sample.sample)):
        fractions = d_spectra_sample.sample[i]
        sigs = generate_signals(1.0, b_values, diffusivities, fractions)
        print(sigs)
