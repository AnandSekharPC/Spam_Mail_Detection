
# --- Custom Naive Bayes ---
class NaiveBayesModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize log-prior and log-likelihood probabilities
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        # Loop over each class and compute the likelihood and prior
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            total_word_count = X_c.sum(axis=0) + self.alpha
            total_class_word_count = total_word_count.sum()
            self.feature_log_prob_[idx, :] = np.log(total_word_count / total_class_word_count)

    def predict(self, X):
        log_probs = (X @ self.feature_log_prob_.T) + self.class_log_prior_
        return np.argmax(log_probs, axis=1)

