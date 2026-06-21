"""Python wrapper around R's ``gllvm`` package (Niku et al.).

R's ``gllvm`` is the standard maximum-likelihood / variational baseline for
generalised latent variable models.  This module runs it in a subprocess (via
``Rscript``), exchanging data as CSV through a shared working directory, and
returns the estimated loadings as a numpy array — so the R baseline can be
called with roughly the same ergonomics as the Python fitters in this package.

Environment
-----------
The defaults target a WSL2 setup that drives the *Windows* R install:
``Rscript.exe`` lives under ``/mnt/c/...`` and the CSV exchange directory must
be visible to **both** WSL and Windows (hence also under ``/mnt/c/...``).  For a
native Linux/macOS R install, pass ``rscript="Rscript"`` and any writable
``workdir`` — the path translation is a no-op for non ``/mnt`` paths.

Example
-------
>>> from gllvm.r_gllvm import RGllvm
>>> r = RGllvm(method="VA")          # variational approximation
>>> if r.available():
...     fit = r.fit(Y, num_lv=2)     # Y: (n, p) counts
...     fit.loadings.shape           # (p, q)
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from dataclasses import dataclass, field

import numpy as np

__all__ = ["RGllvm", "RGllvmFit"]

# WSL2 → Windows R defaults (override for a native install).
DEFAULT_RSCRIPT = "/mnt/c/Program Files/R/R-4.5.1/bin/Rscript.exe"
DEFAULT_WORKDIR = "/mnt/c/Users/willwhite/AppData/Local/Temp/r_gllvm_bench"


def _wsl_to_win(path: str) -> str:
    """Map a ``/mnt/<drive>/...`` WSL path to ``<DRIVE>:/...`` for Windows R.

    Any path not under ``/mnt/<drive>/`` is returned unchanged, so this is a
    no-op on a native install.
    """
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return f"{path[5].upper()}:/" + path[7:]
    return path


@dataclass
class RGllvmFit:
    """Result of a single ``gllvm`` fit.

    Attributes
    ----------
    loadings : np.ndarray
        ``(p, q)`` loading matrix — ``theta`` scaled by ``sigma.lv`` per column,
        directly comparable to a Python ``GLLVM.wz``.
    intercepts : np.ndarray or None
        ``(p,)`` response intercepts (``fit$params$beta0``), directly comparable
        to a Python ``GLLVM.bias``.  ``None`` only if the R fit did not emit them.
    num_lv, method, family : fit settings echoed back.
    stdout, stderr : captured R output (for debugging).
    """

    loadings: np.ndarray
    num_lv: int
    method: str
    family: str
    intercepts: "np.ndarray | None" = None
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)


class RGllvm:
    """Thin, reusable wrapper around ``gllvm::gllvm()``.

    Parameters
    ----------
    rscript : str
        Path to the ``Rscript`` executable.
    workdir : str
        Directory for CSV/script exchange (created if absent).  On WSL2 this
        must be Windows-visible (under ``/mnt/c/...``).
    method : {"VA", "LA", "EVA"}
        Estimation method passed to ``gllvm``.
    family : str
        Response family (e.g. ``"poisson"``, ``"negative.binomial"``).
    maxit : int
        Optimiser iteration cap.
    starting_val : str
        ``control.start$starting.val`` (``"res"`` is robust and matches the
        original benchmark).
    timeout : float
        Subprocess timeout in seconds.
    """

    def __init__(
        self,
        rscript: str = DEFAULT_RSCRIPT,
        workdir: str = DEFAULT_WORKDIR,
        method: str = "VA",
        family: str = "poisson",
        maxit: int = 2000,
        starting_val: str = "res",
        timeout: float = 600.0,
        ntrials: int = 1,
        link: Optional[str] = None,
    ):
        self.rscript = rscript
        self.workdir = workdir
        self.method = method
        self.family = family
        self.maxit = maxit
        self.starting_val = starting_val
        self.timeout = timeout
        self.ntrials = ntrials       # number of binomial trials (binomial family only)
        self.link = link             # e.g. "logit"/"probit" for binomial; None = gllvm default
        os.makedirs(self.workdir, exist_ok=True)

    def available(self) -> bool:
        """True if the configured ``Rscript`` executable exists."""
        return os.path.exists(self.rscript)

    def _ntrials_arg(self) -> str:
        """``Ntrials=`` argument for binomial fits with >1 trial (else empty)."""
        if self.family == "binomial" and self.ntrials != 1:
            return f"\n                         Ntrials={int(self.ntrials)},"
        return ""

    def _link_arg(self) -> str:
        """``link=`` argument (e.g. ``"logit"`` for binomial; else gllvm default).

        gllvm's binomial default is **probit**; pass ``link="logit"`` to match
        logit-generated data (otherwise recovered loadings differ by the
        logit/probit scale factor ~1.8 = π/√3).
        """
        if self.link is not None:
            return f'\n                         link="{self.link}",'
        return ""

    def _render_script(self, y_win: str, w_win: str, b_win: str,
                       num_lv: int, seed: int) -> str:
        return textwrap.dedent(
            f"""
            set.seed({seed})
            Y <- as.matrix(read.csv("{y_win}", header=FALSE))
            suppressPackageStartupMessages(library(gllvm))
            fit <- gllvm(Y, num.lv={num_lv}, family="{self.family}",
                         method="{self.method}",{self._ntrials_arg()}{self._link_arg()}
                         control=list(TMB=TRUE, maxit={self.maxit}, trace=FALSE),
                         control.start=list(starting.val="{self.starting_val}"))
            W <- sweep(fit$params$theta, 2, fit$params$sigma.lv, "*")
            write.csv(W, "{w_win}", row.names=FALSE)
            write.csv(fit$params$beta0, "{b_win}", row.names=FALSE)
            cat("gllvm-done\\n")
            """
        ).strip()

    def fit(self, Y, num_lv: int, seed: int = 42) -> RGllvmFit:
        """Fit ``gllvm`` to count matrix ``Y`` ``(n, p)`` and return loadings.

        Raises ``RuntimeError`` if R is unavailable or the fit fails (inspect
        the message for R's stderr).
        """
        if not self.available():
            raise RuntimeError(
                f"Rscript not found at {self.rscript!r}; pass rscript=... or "
                "check that R is installed."
            )

        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, p); got shape {Y.shape}")

        tag = f"{self.method}_{self.family}_q{num_lv}"
        y_path = os.path.join(self.workdir, f"Y_{tag}.csv")
        w_path = os.path.join(self.workdir, f"W_{tag}.csv")
        b_path = os.path.join(self.workdir, f"B_{tag}.csv")
        r_path = os.path.join(self.workdir, f"run_{tag}.R")
        for stale in (w_path, b_path):
            if os.path.exists(stale):
                os.remove(stale)  # so a stale file can't be mistaken for success

        if self.family == "poisson":
            np.savetxt(y_path, np.rint(Y).astype(int), delimiter=",", fmt="%d")
        else:
            np.savetxt(y_path, Y, delimiter=",", fmt="%.10g")

        with open(r_path, "w") as f:
            f.write(self._render_script(_wsl_to_win(y_path), _wsl_to_win(w_path),
                                        _wsl_to_win(b_path), num_lv, seed))

        proc = subprocess.run(
            [self.rscript, "--vanilla", _wsl_to_win(r_path)],
            capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0 or not os.path.exists(w_path):
            raise RuntimeError(
                f"R gllvm fit failed (returncode={proc.returncode}).\n"
                f"--- R stderr (tail) ---\n{proc.stderr[-2000:]}"
            )

        W = np.loadtxt(w_path, delimiter=",", skiprows=1)  # skip write.csv header
        if W.ndim == 1:                                     # single latent → column
            W = W.reshape(-1, num_lv)
        b = None
        if os.path.exists(b_path):
            b = np.loadtxt(b_path, delimiter=",", skiprows=1)  # beta0 column
            b = np.atleast_1d(b)
        return RGllvmFit(loadings=W, intercepts=b, num_lv=num_lv,
                         method=self.method, family=self.family,
                         stdout=proc.stdout, stderr=proc.stderr)
