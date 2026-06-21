"""Fetch a ready-to-go 10x Visium spatial-transcriptomics section and summarise it.
Raw counts (Poisson) + 2D spatial coords = exactly the GP-GLLVM setting."""
import scanpy as sc, numpy as np
sc.settings.verbosity = 1
ad = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")
ad.var_names_make_unique()
X = ad.X
print("adata:", ad.shape, "(spots x genes)")
print("spatial coords key:", "spatial" in ad.obsm, ad.obsm["spatial"].shape, ad.obsm["spatial"].dtype)
xy = np.asarray(ad.obsm["spatial"], float)
print("coord extent x:", xy[:,0].min(), xy[:,0].max(), " y:", xy[:,1].min(), xy[:,1].max())
import scipy.sparse as sp
counts = X if not sp.issparse(X) else X
tot = np.asarray(counts.sum(0)).ravel()
mean_per_spot = np.asarray(counts.mean(1)).ravel()
print("is integer counts:", np.allclose(counts.data[:50], np.round(counts.data[:50])) if sp.issparse(counts) else np.allclose(counts.flat[:50], np.round(counts.flat[:50])))
print("median total UMI / spot:", np.median(np.asarray(counts.sum(1)).ravel()))
print("genes:", ad.n_vars, " top-expressed:", ad.var_names[np.argsort(tot)[::-1][:8]].tolist())
# nearest-neighbour spacing of spots (sets the spatial scale for ℓ)
from scipy.spatial import cKDTree
d,_ = cKDTree(xy).query(xy, k=2)
print("nearest-neighbour spot spacing: median", np.median(d[:,1]), "px")
