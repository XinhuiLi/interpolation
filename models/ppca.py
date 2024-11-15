import numpy as np

class PPCA:
    def __init__(self, latent_dim, max_iter=1000, tol=1e-4, seed=0):
        self.latent_dim = latent_dim
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        np.random.seed(seed=self.seed)
        self.W = np.random.randn(n_features, self.latent_dim)
        self.sigma_sq = np.var(X_centered)

        for _ in range(self.max_iter):
            old_sigma_sq = self.sigma_sq

            # E-step
            M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma_sq * np.eye(self.latent_dim))
            E_z = M_inv @ self.W.T @ X_centered.T
            E_zz = self.sigma_sq * M_inv + E_z @ E_z.T

            # M-step
            self.W = X_centered.T @ E_z.T @ np.linalg.inv(E_zz)
            self.sigma_sq = 1/n_features*np.mean(np.sum(X_centered**2, axis=1) - 2*E_z.T @ self.W.T @ X_centered.T + np.trace(E_zz @ self.W.T @ self.W))

            if np.abs(self.sigma_sq - old_sigma_sq) < self.tol:
                break

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.W @ np.linalg.inv(self.W.T @ self.W + self.sigma_sq * np.eye(self.latent_dim))

    def inverse_transform(self, Z):
        return Z @ self.W.T + self.mean