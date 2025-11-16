import torch
import torch.nn as nn
from gllvm.glms import *


class GLLVM(nn.Module):
    def __init__(self, num_latent, num_response, num_covar, bias=True):
        super().__init__()
        self.q = num_latent
        self.p = num_response
        self.k = num_covar

        self.wz = nn.Parameter(torch.randn((self.q, self.p)) * 0.1)
        self.wx = nn.Parameter(torch.randn((self.k, self.p)) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None


# class GLLVM(nn.Module):
#     def __init__(self, num_latent, num_response, num_covar, bias=True):
#         super().__init__()
#         self.q = num_latent
#         self.p = num_response
#         self.k = num_covar

#         self.wz = nn.Parameter(torch.randn((self.q, self.p)) * 0.1)
#         self.wx = nn.Parameter(torch.randn((self.k, self.p)) * 0.1)
#         self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None

#         self.multiglm = MultiGLM(dim=num_response)
#         self.register_glm = self.multiglm.register_glm

#     def forward(self, z, x=None):
#         """
#         Compute the conditional mean of a GLLVM

#         Parameters:
#             - z: the latent variables, of shape (num_obs, num_latent)
#             - x: covariates. If not None, must be of shape (num_obs, num_covar)
#         """
#         linpar = self.compute_linpar(z, x)
#         mean = self.multiglm(linpar)

#         return linpar, mean

#     def compute_linpar(self, z, x):

#         # Input checks
#         if z.shape[1] != self.q:
#             raise ValueError(
#                 f"Expected to receive latent variables z of shape (num_obs, {self.q})."
#             )

#         if x is None and self.k > 0:
#             raise ValueError(
#                 f"Expected to receive covariates x but received None instead."
#             )

#         # Computing linpar
#         # ----------------
#         linpar = z @ self.wz

#         if x is not None:
#             linpar += x @ self.wx

#         if self.bias is not None:
#             linpar += self.bias

#         return linpar

#     def sample(self, num_obs=None, z=None):
#         if z is None:
#             z = torch.randn((num_obs, self.q), device=next(self.parameters()).device)

#         linpar, mean = self(z)
#         return self.multiglm.sample(mean)
