import numpy as np
import matplotlib.pyplot as plt

N = 100  # Number of spins
spins = np.random.choice([-1, 1], size=N)  # Initialize spins randomly

J = 1  # Interaction strength

def ising_hamiltonian(spins, J):
    return -J * np.sum(spins[:-1] * spins[1:])

energy = ising_hamiltonian(spins, J)
print("Initial Energy of the system:", energy)

plt.figure(figsize=(10, 2))
plt.scatter(range(N), spins, c=spins, cmap='bwr', marker='_', lw=2)
plt.ylim(-1.5, 1.5)
plt.title('Initial Random Spin Configuration')
plt.savefig('spin_configuration.png')
