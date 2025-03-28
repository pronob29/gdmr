# dmr.py

import numpy as np
from scipy.special import gammaln, digamma
import scipy.optimize as optimize
import pickle
from functools import lru_cache
from scipy.special import logsumexp
from collections import defaultdict
from logging import getLogger
from lda import LDA  # Ensure correct import based on your project structure

class DMR(LDA):
    '''
    Topic Model with Dirichlet Multinomial Regression
    '''
    def __init__(self, G, sigma, beta, docs, vecs, V, trained=None, target_number=100):
        super(DMR, self).__init__(G, 0.01, beta, docs, V, trained)
        self.L = vecs.shape[1]  # Feature vector length (dimensionality)
        self.vecs = vecs  # Document feature vectors
        self.sigma = sigma  # Sigma parameter for the normal distribution
        self.target_number = target_number  # Target iteration for switching to BFGS
        self.objective_values = []  # List to store objective values
        self.perplexity_scores = []  # List to store perplexity scores

        # Initialize Lambda with a multivariate normal distribution
        self.Lambda = np.random.multivariate_normal(
            mean=np.zeros(self.L),
            cov=(self.sigma ** 2) * np.identity(self.L),
            size=self.G
        )

        # Calculate the alpha parameter
        if trained is None:
            self.alpha = self.get_alpha()
        else:
            # Using parameters from a trained model
            self.Lambda = trained.Lambda
            self.alpha = trained.alpha

        # Initialize logger
        self.logger = getLogger(self.__class__.__name__)

    @lru_cache(maxsize=1000)
    def get_alpha_cached(self, Lambda_bytes):
        """
        Calculate alpha values using cached Lambda parameters.
        """
        # Deserialize the byte strings back to NumPy arrays
        Lambda = pickle.loads(Lambda_bytes)

        # Compute Lambda dot vecs
        linear = self.vecs.dot(Lambda.T)  # Shape: (N, G)

        # To ensure numerical stability, clip the values to prevent overflow
        linear = np.clip(linear, a_min=-20, a_max=20)

        # Compute alpha = exp(Lambda^T x)
        alpha = np.exp(linear)  # Shape: (N, G)

        return alpha

    def get_alpha(self, Lambda=None):
        """
        alpha = exp(Lambda^T x)
        """
        if Lambda is None:
            Lambda = self.Lambda if self.trained is None else self.trained.Lambda

        # Serialize Lambda to byte strings for caching
        Lambda_bytes = pickle.dumps(Lambda)

        return self.get_alpha_cached(Lambda_bytes)

    def training(self, iteration, voca, burn_in=100):
        """
        Perform DMR model training with Gibbs sampling and BFGS optimization.

        :param iteration: Total number of iterations for the training.
        :param voca: Vocabulary for the corpus, used for outputting word distributions.
        :param burn_in: Number of iterations to run after switching to BFGS optimization before recalculating alpha.
        """
        bfgs_started = False  # Flag to indicate whether BFGS optimization has started
        update_topics = True  # Control flag for updating topics

        print(f"Target iteration for BFGS start: {self.target_number + 1}")
        for i in range(iteration):
            print(f"Starting iteration {i + 1}/{iteration}")

            if i < self.target_number:
                self.inference(update_topics=update_topics)  # Perform Gibbs sampling
            else:
                if not bfgs_started:
                    print(f"Reached target iteration {self.target_number + 1}. Starting BFGS optimization.")
                    bfgs_started = True
                    update_topics = False  # Stop updating topics

                # BFGS optimization without updating topics
                self.bfgs()

                if i >= self.target_number + burn_in:
                    self.alpha = self.get_alpha()  # Recalculate alpha values

            # Calculate and log perplexity at specified intervals
            if (i + 1) % self.SAMPLING_RATE == 0 or i == iteration - 1:
                perp = self.perplexity()
                self.logger.info(f"Perplexity at iteration {i + 1}: {perp}")
                self.perplexity_scores.append(perp) 

        # Output word distributions after training
        self.output_word_dist_with_voca(voca)

    def bfgs(self):
        """
        BFGS optimization for Lambda.
        """
        self.objective_values = []  # Reset the logging

        def ll(params):
            x = params.reshape((self.G, self.L))
            obj_value = self._ll(x)
            self.objective_values.append(obj_value)  # Log objective values
            return obj_value

        def dll(params):
            x = params.reshape((self.G, self.L))
            result = self._dll(x)
            return result.ravel()

        # Initialize Lambda for optimization
        initial_Lambda = np.random.multivariate_normal(
            mean=np.zeros(self.L),
            cov=(self.sigma ** 2) * np.identity(self.L),
            size=self.G
        )
        initial = initial_Lambda.ravel()

        # Define bounds: Lambda can be any real number
        bounds = [(None, None)] * (self.G * self.L)

        # Optimize Lambda using BFGS
        newLambda, fmin, res = optimize.fmin_l_bfgs_b(ll, initial, dll, bounds=bounds)
        self.Lambda = newLambda.reshape((self.G, self.L))

    def _ll(self, x):
        """
        Log-likelihood calculation.
        """
        # Calculate alpha
        alpha = self.get_alpha(x)

        # Initialize result
        result = 0.0

        # P(w|z)
        result += self.G * gammaln(self.beta * self.G)
        result += -np.sum(gammaln(np.sum(self.n_z_w, axis=1)))
        result += np.sum(gammaln(self.n_z_w))
        result += -self.V * gammaln(self.beta)

        # P(z|Lambda)
        result += np.sum(gammaln(np.sum(alpha, axis=1)))
        result += -np.sum(gammaln(np.sum(self.n_m_z + alpha, axis=1)))
        result += np.sum(gammaln(self.n_m_z + alpha))
        result += -np.sum(gammaln(alpha))

        # P(Lambda)
        result += -self.G / 2.0 * np.log(2.0 * np.pi * (self.sigma ** 2))
        result += -1.0 / (2.0 * (self.sigma ** 2)) * np.sum(x ** 2)

        # Return negative log-likelihood for minimization
        return -result

    def _dll(self, x):
        """
        Gradient of the log-likelihood.
        """
        alpha = self.get_alpha(x)
        exp_product = self.get_exp_product(x)

        # Adding a small constant to alpha to ensure positivity
        alpha += 1e-4

        # Gradient with respect to Lambda
        dLambda = np.sum(
            self.vecs[:, np.newaxis, :] * exp_product[:, :, np.newaxis] *
            (digamma(np.sum(alpha, axis=1))[:, np.newaxis, np.newaxis]
             - digamma(np.sum(self.n_m_z + alpha, axis=1))[:, np.newaxis, np.newaxis]
             + digamma(self.n_m_z + alpha)[:, :, np.newaxis]
             - digamma(alpha)[:, :, np.newaxis]),
            axis=0
        ) - x / (self.sigma ** 2)

        # Handle potential NaNs or Infs
        dLambda = np.nan_to_num(dLambda)

        # Return gradients for minimization
        return dLambda

    def get_exp_product(self, x):
        """
        Compute exp(Lambda.dot(vecs.T)) using numerical stability.
        """
        linear = self.vecs.dot(x.T)  # Shape: (N, G)
        linear = np.clip(linear, a_min=-20, a_max=20)  # Prevent overflow
        exp_product = np.exp(linear)  # Shape: (N, G)
        return exp_product

    def word_dist_with_voca(self, voca, topk=None):
        '''
        Output the word probability of each topic
        '''
        phi = self.worddist()
        if topk is None:
            topk = phi.shape[1]
        result = defaultdict(dict)
        for k in range(self.G):
            for w in np.argsort(-phi[k])[:topk]:
                result[k][voca[w]] = phi[k, w]
        return result

    def output_word_dist_with_voca(self, voca, topk=10):
        word_dist = self.word_dist_with_voca(voca, topk)
        for k in word_dist:
            word_dist[k] = sorted(word_dist[k].items(), key=lambda x: x[1], reverse=True)
            for w, v in word_dist[k]:
                self.log(self.logger.debug, "TOPIC", [k, w, v])

    def log(self, method, etype, messages):
        method("\t".join(map(str, [self.params(), etype] + messages)))

    def params(self):
        return '''G=%d, sigma=%s, beta=%s''' % (self.G, self.sigma, self.beta)

    def __getstate__(self):
        '''
        Logger cannot be serialized
        '''
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        '''
        Logger cannot be serialized
        '''
        self.__dict__.update(state)
        self.logger = getLogger(self.__class__.__name__)
