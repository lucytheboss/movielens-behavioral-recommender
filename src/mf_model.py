
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm


class MatrixFactorization:
    """
    Collaborative Filtering via Matrix Factorization with SGD.

    R ≈ μ + b_u + b_i + P × Q^T

    μ   : global mean
    b_u : user bias  (n_users,)
    b_i : item bias  (n_items,)
    P   : (n_users, n_factors) — user latent factors
    Q   : (n_items, n_factors) — item latent factors

    Loss (L2 regularization):
        L = Σ (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        lambda_: float = 0.05,
        n_epochs: int = 50,
        patience: int = 5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_factors  : latent factor count. lower = less overfitting
        lr         : learning rate
        lambda_    : L2 regularization strength. higher = less overfitting
        n_epochs   : max training epochs
        patience   : early stopping — stop after this many epochs with no val improvement
        """
        self.n_factors = n_factors
        self.lr = lr
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

        self.P: Optional[np.ndarray] = None   # (n_users, n_factors)
        self.Q: Optional[np.ndarray] = None   # (n_items, n_factors)
        self.b_u: Optional[np.ndarray] = None  # (n_users,)
        self.b_i: Optional[np.ndarray] = None  # (n_items,)
        self.mu: float = 0.0                   # global mean

        self.train_losses: list = []
        self.val_losses: list = []
        self.best_epoch: int = 0

        self.user2idx: dict = {}
        self.item2idx: dict = {}
        self.idx2user: dict = {}
        self.idx2item: dict = {}

    def _build_mappings(self, ratings: np.ndarray):
        unique_users = np.unique(ratings[:, 0]).astype(int)
        unique_items = np.unique(ratings[:, 1]).astype(int)
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        return len(unique_users), len(unique_items)

    def fit(
        self,
        train_ratings: np.ndarray,
        val_ratings: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        train_ratings : (N, 3) array — [user_id, item_id, rating]
        val_ratings   : (M, 3) array — used for early stopping
        verbose       : print loss each epoch
        """
        np.random.seed(self.random_state)

        n_users, n_items = self._build_mappings(train_ratings)

        self.mu = float(np.mean(train_ratings[:, 2]))

        self.P = np.random.normal(scale=0.01, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=0.01, size=(n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        train_idx = self._to_idx(train_ratings)

        best_val_loss = np.inf
        epochs_no_improve = 0
        best_P, best_Q, best_b_u, best_b_i = None, None, None, None

        for epoch in range(self.n_epochs):
            np.random.shuffle(train_idx)
            train_loss = self._run_epoch(train_idx)
            self.train_losses.append(train_loss)

            if val_ratings is not None:
                val_idx = self._to_idx(val_ratings, skip_unknown=True)
                val_loss = self._compute_rmse(val_idx)
                self.val_losses.append(val_loss)

                if verbose:
                    print(f"Epoch {epoch+1:>3}/{self.n_epochs} | "
                          f"Train RMSE: {train_loss:.4f} | Val RMSE: {val_loss:.4f}")

                # early stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self.best_epoch = epoch + 1
                    best_P = self.P.copy()
                    best_Q = self.Q.copy()
                    best_b_u = self.b_u.copy()
                    best_b_i = self.b_i.copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1} "
                                  f"(best val RMSE {best_val_loss:.4f} at epoch {self.best_epoch})")
                        self.P, self.Q = best_P, best_Q
                        self.b_u, self.b_i = best_b_u, best_b_i
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch+1:>3}/{self.n_epochs} | Train RMSE: {train_loss:.4f}")

        return self

    def _run_epoch(self, ratings_idx: np.ndarray) -> float:
        total_sq_error = 0.0
        for u, i, r in ratings_idx:
            u, i = int(u), int(i)
            r_hat = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
            e = r - r_hat
            total_sq_error += e ** 2

            p_u_old = self.P[u].copy()

            self.b_u[u] += self.lr * (e - self.lambda_ * self.b_u[u])
            self.b_i[i] += self.lr * (e - self.lambda_ * self.b_i[i])
            self.P[u]   += self.lr * (e * self.Q[i]   - self.lambda_ * self.P[u])
            self.Q[i]   += self.lr * (e * p_u_old     - self.lambda_ * self.Q[i])

        return np.sqrt(total_sq_error / len(ratings_idx))

    def predict(self, user_id: int, item_id: int) -> float:
        u = self.user2idx.get(user_id)
        i = self.item2idx.get(item_id)
        if u is None or i is None:
            return self.mu
        return float(self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i])

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list:
        u = self.user2idx.get(user_id)
        if u is None:
            return []

        scores = self.mu + self.b_u[u] + self.b_i + self.P[u] @ self.Q.T  # (n_items,)

        if exclude_seen and hasattr(self, '_seen'):
            seen = self._seen.get(user_id, set())
            for item_id in seen:
                i = self.item2idx.get(item_id)
                if i is not None:
                    scores[i] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.idx2item[i], float(scores[i])) for i in top_indices]

    def _compute_rmse(self, ratings_idx: np.ndarray) -> float:
        sq_errors = []
        for u, i, r in ratings_idx:
            u, i = int(u), int(i)
            r_hat = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
            sq_errors.append((r - r_hat) ** 2)
        return np.sqrt(np.mean(sq_errors))

    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.train_losses, label='Train RMSE', color='steelblue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val RMSE', color='coral')
            if self.best_epoch:
                plt.axvline(self.best_epoch - 1, color='gray', linestyle='--',
                            label=f'Best epoch ({self.best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Matrix Factorization — Learning Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../images/loss_curve.png', dpi=150)
        plt.show()

    def _to_idx(self, ratings: np.ndarray, skip_unknown: bool = False) -> np.ndarray:
        result = []
        for row in ratings:
            u_raw, i_raw, r = int(row[0]), int(row[1]), float(row[2])
            u = self.user2idx.get(u_raw)
            i = self.item2idx.get(i_raw)
            if u is None or i is None:
                if skip_unknown:
                    continue
                raise ValueError(f"Unknown user {u_raw} or item {i_raw}. Call fit() first.")
            result.append([u, i, r])
        return np.array(result)
