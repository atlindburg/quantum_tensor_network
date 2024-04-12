import qutip as qt

# Define Pauli spin matrices
sigma_x = qt.sigmax()
sigma_y = qt.sigmay()
sigma_z = qt.sigmaz()

# Define the ground and excited states
ground_state = qt.basis(2, 0)
excited_state = qt.basis(2, 1)

# Print states
print("Ground State:\n", ground_state)
print("Excited State:\n", excited_state)

