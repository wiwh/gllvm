import torch
import torch.distributions as D
import matplotlib.pyplot as plt

# Synthetic data
y = torch.tensor([1.5])
mu = torch.nn.Parameter(torch.tensor([0.0]))

# Two different scales
scales = [0.1, 10.0]
grads = []

for s in scales:
    dist = D.Normal(loc=mu, scale=torch.tensor(s))
    logp = dist.log_prob(y)
    logp.backward()  # compute ∂ log p / ∂ mu
    grads.append(mu.grad.item())
    mu.grad.zero_()

# Plot results
plt.figure(figsize=(5, 3))
plt.bar(["σ=0.1 (tight)", "σ=10 (wide)"], grads, color=["#1f77b4", "#ff7f0e"])
plt.ylabel("Gradient magnitude on μ")
plt.title("Effect of scale (σ) on gradient strength")
plt.show()

print(f"Gradient with σ=0.1: {grads[0]:.3f}")
print(f"Gradient with σ=10.0: {grads[1]:.3f}")
