import numpy as np
from scipy import special

# Model Parameters
# 		- Decrease tol (e.g., 1e-6 to 1e-5) for more precise convergence, less strict = faster convergences
#			- Increase max_iter for more training time
#			- Try different random_state values for different initializations
class MixtureMultinomialEM:
    def __init__(self, n_topics, max_iter=100, tol=1e-4, random_state=None):
        self.K = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.vocab_size = None
        self.log_likelihood_history = []
        if random_state is not None:
            np.random.seed(random_state)

    def initialize_parameters(self, n_docs, vocab_size):
        """Initialize model parameters."""
        self.vocab_size = vocab_size
        # Initialize topic proportions
        self.pi = np.random.dirichlet(np.ones(self.K))
        # Initialize word distributions for each topic
        self.mu = np.random.dirichlet(np.ones(vocab_size), size=self.K)

    def e_step(self, T):
        """Perform E-step: compute responsibilities."""
        n_docs = T.shape[0]
        # Compute log probabilities
        # Use scipy.special functions for numerical stability
        log_pi = np.log(self.pi + 1e-100)
        log_mu = np.log(self.mu + 1e-100)

        # Compute log likelihood for each doc-topic pair
        log_gamma = np.zeros((n_docs, self.K))
        for k in range(self.K):
            log_gamma[:, k] = np.dot(T, log_mu[k]) + log_pi[k]

        # Normalize (in log space)
        # Use logsumexp for numerical stability
        log_gamma = log_gamma - special.logsumexp(log_gamma, axis=1)[:, np.newaxis]
        # Convert back from log space
        gamma = np.exp(log_gamma)

        # Compute log likelihood
        log_likelihood = np.sum(special.logsumexp(log_gamma, axis=1))
        self.log_likelihood_history.append(log_likelihood)

        return gamma

    def m_step(self, T, gamma):
        """Perform M-step: update parameters."""
        # Update pi
        # Use softmax for numerical stability when needed
        self.pi = special.softmax(np.log(gamma.mean(axis=0) + 1e-100))

        # Update mu
        denominator = gamma.sum(axis=0)[:, np.newaxis]
        self.mu = np.dot(gamma.T, T) / denominator

    def fit(self, T):
        """Fit the model to the data."""
        n_docs, vocab_size = T.shape
        self.initialize_parameters(n_docs, vocab_size)

        for iteration in range(self.max_iter):
            # E-step
            gamma = self.e_step(T)

            # M-step
            old_mu = self.mu.copy()
            self.m_step(T, gamma)

            # Check convergence
            if iteration > 0:
                diff = np.max(np.abs(self.mu - old_mu))
                if diff < self.tol:
                    print(f"Converged after {iteration} iterations")
                    break

        return self

    def get_top_words(self, vocab, n_words=10):
        """Get the top n words for each topic."""
        topics = []
        for k in range(self.K):
            top_idx = np.argsort(-self.mu[k])[:n_words]
            topics.append([vocab[i] for i in top_idx])
        return topics

    def predict(self, T):
        """Predict topic distributions for new documents."""
        gamma = self.e_step(T)
        return gamma
