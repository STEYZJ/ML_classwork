from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors


@dataclass
class SelectionResult:
    ranking: np.ndarray
    scores: np.ndarray
    objectives: list[float]
    n_iter: int


def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, copy=False)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mean) / std


def build_knn_graph(X: np.ndarray, n_neighbors: int = 5) -> sp.csr_matrix:
    n_samples = X.shape[0]
    n_neighbors = min(max(1, n_neighbors), n_samples - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    positive = distances[distances > 0]
    sigma = np.median(positive) if positive.size else 1.0
    sigma = max(float(sigma), 1e-12)
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))

    rows = np.repeat(np.arange(n_samples), n_neighbors)
    graph = sp.coo_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(n_samples, n_samples),
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def laplacian_from_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    return sp.diags(degree) - graph


def spectral_targets(graph: sp.csr_matrix, n_components: int, seed: int = 42) -> np.ndarray:
    n_components = max(1, min(n_components, graph.shape[0] - 1))
    try:
        Y = spectral_embedding(
            graph,
            n_components=n_components,
            random_state=seed,
            drop_first=False,
            norm_laplacian=True,
        )
    except Exception:
        dense = graph.toarray()
        degree = dense.sum(axis=1)
        lap = np.diag(degree) - dense
        values, vectors = np.linalg.eigh(lap)
        order = np.argsort(values)[:n_components]
        Y = vectors[:, order]
    Y = np.asarray(Y, dtype=np.float64)
    Y -= Y.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Y, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return Y / norms


def graph_sparse_regression_ranking(
    X: np.ndarray,
    graph: sp.csr_matrix,
    n_components: int,
    alpha: float = 1e-2,
    beta: float = 1e-1,
    max_iter: int = 15,
    tol: float = 1e-4,
    ridge: float = 1e-6,
    cg_max_iter: int = 300,
    seed: int = 42,
) -> SelectionResult:
    """Rank features by graph-regularized sparse regression.

    The optimization problem is
        min_W ||XW - Y||_F^2 + alpha * ||W||_2,1
              + beta * Tr(W^T X^T L X W),
    where Y is a spectral embedding of the kNN graph.
    """
    X = np.asarray(X, dtype=np.float64)
    L = laplacian_from_graph(graph)
    Y = spectral_targets(graph, n_components=n_components, seed=seed)
    n_features = X.shape[1]
    n_components = Y.shape[1]

    W = np.zeros((n_features, n_components), dtype=np.float64)
    row_weights = np.ones(n_features, dtype=np.float64)
    objectives: list[float] = []

    for iteration in range(1, max_iter + 1):
        W = _solve_weight_matrix(
            X=X,
            L=L,
            Y=Y,
            row_weights=row_weights,
            alpha=alpha,
            beta=beta,
            ridge=ridge,
            previous=W,
            cg_max_iter=cg_max_iter,
        )
        row_norms = np.linalg.norm(W, axis=1)
        row_weights = 1.0 / (2.0 * row_norms + 1e-12)
        obj = _objective(X, L, Y, W, alpha, beta)
        objectives.append(obj)
        if iteration > 1:
            prev = max(abs(objectives[-2]), 1e-12)
            if abs(objectives[-2] - objectives[-1]) / prev < tol:
                break

    scores = np.linalg.norm(W, axis=1)
    ranking = np.argsort(-scores)
    return SelectionResult(ranking=ranking, scores=scores, objectives=objectives, n_iter=len(objectives))


def variance_ranking(X: np.ndarray) -> np.ndarray:
    return np.argsort(-np.var(X, axis=0))


def laplacian_score_ranking(X: np.ndarray, graph: sp.csr_matrix) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    L = laplacian_from_graph(graph)
    denom_degree = float(degree.sum())
    if denom_degree <= 1e-12:
        return variance_ranking(X)

    weighted_mean = (degree @ X) / denom_degree
    centered = X - weighted_mean
    numerator = np.asarray(centered.T @ (L @ centered)).diagonal()
    denominator = np.sum(centered * (degree[:, None] * centered), axis=0)
    invalid = denominator < 1e-12
    safe_denominator = denominator.copy()
    safe_denominator[invalid] = 1.0
    score = numerator / safe_denominator
    score[invalid] = np.inf
    return np.argsort(score)


def _solve_weight_matrix(
    X: np.ndarray,
    L: sp.csr_matrix,
    Y: np.ndarray,
    row_weights: np.ndarray,
    alpha: float,
    beta: float,
    ridge: float,
    previous: np.ndarray,
    cg_max_iter: int,
) -> np.ndarray:
    n_features = X.shape[1]
    B = X.T @ Y

    if n_features <= 1200:
        A = X.T @ X
        if beta > 0:
            A = A + beta * (X.T @ (L @ X))
        A = A + np.diag(alpha * row_weights + ridge)
        return np.linalg.solve(A, B)

    def matvec(v: np.ndarray) -> np.ndarray:
        xv = X @ v
        out = X.T @ xv
        if beta > 0:
            out = out + beta * (X.T @ (L @ xv))
        out = out + (alpha * row_weights + ridge) * v
        return out

    operator = LinearOperator((n_features, n_features), matvec=matvec, dtype=np.float64)
    W = np.empty((n_features, Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        solution, _ = cg(
            operator,
            B[:, j],
            x0=previous[:, j],
            rtol=1e-4,
            atol=1e-7,
            maxiter=cg_max_iter,
        )
        W[:, j] = solution
    return W


def _objective(
    X: np.ndarray,
    L: sp.csr_matrix,
    Y: np.ndarray,
    W: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    embedding = X @ W
    loss = np.linalg.norm(embedding - Y, ord="fro") ** 2
    sparse = alpha * np.linalg.norm(W, axis=1).sum()
    manifold = beta * float(np.trace(embedding.T @ (L @ embedding)))
    return float(loss + sparse + manifold)
