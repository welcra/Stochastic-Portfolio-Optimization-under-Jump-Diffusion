import numpy as np
import matplotlib.pyplot as plt

def simulate_jd(S0, mu, sigma, lamb, muJ, sigmaJ, T, N, M):
    dt = T / N
    prices = np.zeros((M, N + 1))
    prices[:, 0] = S0
    
    for t in range(1, N + 1):
        Z = np.random.normal(size=M)
        jump = np.random.poisson(lamb * dt, M)
        J = np.random.lognormal(mean=muJ, sigma=sigmaJ, size=M)
        
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        ) * (J**jump)
    
    return prices

S0 = 100
mu = 0.1
sigma = 0.2
lamb = 0.3
muJ = -0.4
sigmaJ = 0.3
T = 1
N = 252
M = 10

paths = simulate_jd(S0, mu, sigma, lamb, muJ, sigmaJ, T, N, M)
for i in range(M):
    plt.plot(paths[i])
plt.title("Jump-Diffusion Asset Paths")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()


def utility(w, paths, S0, W0=1):
    ST = paths[:, -1]
    WT = W0 * (w * ST / S0 + (1 - w))
    return np.mean(np.log(WT))

weights = np.linspace(0, 1, 100)
utils = [utility(w, paths, S0) for w in weights]

optimal_w = weights[np.argmax(utils)]
print(f"Optimal weight in risky asset: {optimal_w:.4f}")

plt.plot(weights, utils)
plt.title("Expected Log Utility vs Allocation")
plt.xlabel("Weight in risky asset")
plt.ylabel("Expected log utility")
plt.axvline(optimal_w, color='red', linestyle='--')
plt.show()