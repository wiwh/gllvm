"""Preprocess the Visium section into a compact GP-GLLVM problem and cache to npz.
 - keep RAW integer counts (Poisson) for P spatially/highly-variable genes
 - drop mito + Bc1 (depth/QC artefacts)
 - coords rescaled to spot-pitch units (1 unit = nearest-neighbour spacing ≈ 100 µm)
 - per-spot offset = log(total UMI / median)  -> factors model structure, not library size
"""
import scanpy as sc, numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree

P = 100
ad = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")
ad.var_names_make_unique()
sc.pp.filter_genes(ad, min_cells=int(0.10 * ad.n_obs))           # expressed in >=10% spots
drop = ad.var_names.str.startswith(("mt-", "Mt-")) | (ad.var_names == "Bc1")
ad = ad[:, ~drop].copy()

raw = ad.X.copy()                                                  # raw counts (spots x genes)
lib = np.asarray(raw.sum(1)).ravel().astype(float)
offset = np.log(lib / np.median(lib))

# pick highly-variable genes on log-normalised data, but KEEP their raw counts
norm = ad.copy(); sc.pp.normalize_total(norm, target_sum=1e4); sc.pp.log1p(norm)
sc.pp.highly_variable_genes(norm, n_top_genes=P, flavor="seurat")
hv = norm.var["highly_variable"].values
Y = np.asarray(raw[:, hv].todense() if sp.issparse(raw) else raw[:, hv]).astype(np.float64)
genes = ad.var_names[hv].tolist()

xy = np.asarray(ad.obsm["spatial"], float)
xy = xy - xy.mean(0)
pitch = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
xy = xy / pitch                                                    # 1 unit = 1 spot pitch (~100 µm)

np.savez_compressed("/home/willwhite/GitHub/gllvm/playground/gp-gllvm/_visium.npz",
                    Y=Y, xy=xy, offset=offset, genes=np.array(genes), pitch=pitch)
print(f"saved _visium.npz  Y={Y.shape} (spots x genes)  coords range={xy.min():.1f}..{xy.max():.1f} spot-units")
print(f"pitch={pitch:.1f}px (=1 unit≈100µm)  mean count/gene/spot={Y.mean():.2f}  median UMI/spot={np.median(lib):.0f}")
print("genes[:12]:", genes[:12])
