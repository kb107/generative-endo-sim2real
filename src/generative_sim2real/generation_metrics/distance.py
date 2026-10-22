"""Set-to-set distances on feature matrices and on pixel statistics (torch only, no models).

KID: unbiased squared MMD with the cubic polynomial kernel k(x,y) = (x.y/d + 1)^3
(Binkowski et al. 2018), computed on all samples and, for the usual variance estimate, on random
equal-size subsets. Frechet distance: ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2)) with the trace
term from the eigenvalues of S1^(1/2) S2 S1^(1/2) (symmetric, stable). With n < d the covariances
are rank deficient and the Frechet value is biased upward and depends on n: compare only at equal n.
"""
import torch

from generative_sim2real.generation.p4_report import wasserstein_1d


def _poly_kernel(a, b):
    return (a @ b.T / a.shape[1] + 1.0) ** 3


def mmd2_unbiased(x, y):
    """Unbiased MMD^2 (U-statistic; the within-set diagonal is excluded)."""
    x, y = x.double(), y.double()
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError('MMD needs at least two samples per set')
    kxx, kyy, kxy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
    term_x = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    term_y = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return (term_x + term_y - 2 * kxy.mean()).item()


def kid(x, y, *, subsets=50, subset_size=100, seed=0):
    """(full-sample unbiased MMD^2, mean over random subsets, std over subsets, subset size)."""
    full = mmd2_unbiased(x, y)
    size = min(subset_size, len(x), len(y))
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(subsets):
        i = torch.randperm(len(x), generator=generator)[:size]
        j = torch.randperm(len(y), generator=generator)[:size]
        values.append(mmd2_unbiased(x[i], y[j]))
    values = torch.tensor(values, dtype=torch.float64)
    return {'full': full, 'subset_mean': values.mean().item(),
            'subset_std': values.std().item() if len(values) > 1 else 0.0,
            'subset_size': size, 'subsets': subsets}


def _mean_cov(x):
    x = x.double()
    mean = x.mean(0)
    centered = x - mean
    return mean, centered.T @ centered / max(len(x) - 1, 1)


def frechet_distance(x, y):
    """Frechet distance between Gaussians fitted to two feature sets (squared, as in FID)."""
    if len(x) < 2 or len(y) < 2:
        raise ValueError('Frechet distance needs at least two samples per set')
    mu1, s1 = _mean_cov(x)
    mu2, s2 = _mean_cov(y)
    values, vectors = torch.linalg.eigh(s1)
    root = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    middle = root @ s2 @ root
    eigen = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0)
    trace = s1.trace() + s2.trace() - 2 * eigen.sqrt().sum()
    return ((mu1 - mu2).pow(2).sum() + trace).item()


def channel_w1(a, b, *, quantiles=1000):
    """Per-channel 1-D Wasserstein-1 between pooled pixel intensities ([N,3,H,W] in [0,1]) and
    its mean over R,G,B (same statistic as p4_report.rgb_w1)."""
    per = [wasserstein_1d(a[:, c], b[:, c], quantiles=quantiles) for c in range(3)]
    return {'R': per[0], 'G': per[1], 'B': per[2], 'mean': sum(per) / 3}


def pairwise_l1(flat):
    """Mean pairwise mean-absolute difference of flattened images [N,D] (same definition as
    p4_report.pairwise_diversity, vectorised)."""
    count = len(flat)
    if count < 2:
        raise ValueError('Diversity needs at least two images')
    distance = torch.cdist(flat.double(), flat.double(), p=1) / flat.shape[1]
    return (distance.sum() / (count * (count - 1))).item()
