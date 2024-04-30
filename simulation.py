import numpy as np

def initialize_mps(N, d=2):
    # Initialize random tensors for MPS
    # Each tensor will have dimensions (d, chi, chi) where chi is the bond dimension
    # For the edge tensors, the bond dimension on one side should be 1
    chi = 4  # A small initial bond dimension
    mps = [np.random.rand(d, 1, chi)]  # First tensor
    for i in range(1, N - 1):
        mps.append(np.random.rand(d, chi, chi))
    mps.append(np.random.rand(d, chi, 1))  # Last tensor
    return mps

def construct_mpo(N, J, h):
    # Constructing an MPO for the Ising model
    # Here we need to define the MPO tensors that apply the Hamiltonian terms
    sz = np.array([[1, 0], [0, -1]])  # Pauli Z
    sx = np.array([[0, 1], [1, 0]])   # Pauli X
    identity = np.eye(2)
    mpo = []
    # Use the Kronecker product to construct MPO tensors
    for i in range(N):
        mpo_tensor = np.zeros((2, 2, 4, 4))
        mpo_tensor[:, :, 0, 0] = identity
        mpo_tensor[:, :, 0, 1] = sz
        mpo_tensor[:, :, 0, 2] = sx * h
        mpo_tensor[:, :, 1, 3] = sz * J
        mpo_tensor[:, :, 3, 3] = identity
        mpo.append(mpo_tensor)
    return mpo

import numpy as np

def optimize_tensor(tensor, neighbors, mpo, is_edge):
    left, right = neighbors
    
    # Debug statement to see where we are at before beginning.
    print(f"Optimizing tensor with left shape {left.shape}, right shape {right.shape}, and MPO shape {mpo.shape}")
    
    if is_edge:
        print("Processing an edge tensor.")

    # Assuming you need to adjust the MPO tensor shape for correct contraction:
    # The goal is to ensure dimensions of left.shape[2] and mpo.shape[2] match for contraction
    mpo_adjusted = mpo.transpose(2, 3, 0, 1)  # This changes the MPO configuration to (4, 4, 2, 2)
    #mpo_adjusted = mpo.transpose(2, 0, 1, 3)


    # Check dimensions before contracting
    if left.shape[2] != mpo_adjusted.shape[0]:
        print("Mismatch in dimensions between left tensor and MPO:")
        print(f"Left dimension: {left.shape[2]}, MPO dimension: {mpo_adjusted.shape[0]}")
        # Adjust the logic or raise an error
        raise ValueError("Dimension mismatch for contraction")

    try:
        # Contract the left tensor with the MPO
        # Axes: Last of left (bond dimension) with first of MPO
        left_mpo = np.tensordot(left, mpo_adjusted, axes=([2], [0]))
        print("Shape after contracting left with MPO:", left_mpo.shape)

        # Now contract this result with the right tensor
        # Axes: Last of the result (from MPO) with second of right (bond dimension)
        local_hamiltonian = np.tensordot(left_mpo, right, axes=([2], [1]))
        print("Local Hamiltonian shape:", local_hamiltonian.shape)

    except ValueError as e:
        print("Error during tensor contraction:", e)
        raise
        
    '''
    try:
        left_mpo = np.tensordot(left, mpo_adjusted, axes=([2], [0]))
        print("Shape after contracting Left with MPO:", left_mpo.shape)

        # Determine the correct axes for the final contraction
        final_axes = ([2, 3], [0, 1])  # This is an example; adjust based on your specific dimensions
        local_hamiltonian = np.tensordot(left_mpo, right, axes=final_axes)
        print("Local Hamiltonian Shape:", local_hamiltonian.shape)

    except ValueError as e:
        print("Error during tensor contraction:", e)
        raise
    '''

    # Reshape for SVD
    tensor_shape = tensor.shape
    matrix_form = tensor.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
    
    # Apply SVD
    U, S, Vh = np.linalg.svd(matrix_form, full_matrices=False)
    
    # Truncate the SVD results to retain only the most significant components
    chi = min(len(S), 10)  # Example: keep up to 10 singular values
    U = U[:, :chi]
    S = np.diag(S[:chi])
    Vh = Vh[:chi, :]
    
    # Update the tensor
    tensor = np.dot(U, np.dot(S, Vh)).reshape(tensor_shape)
    
    # Normalize the tensor if necessary
    norm = np.linalg.norm(tensor)
    tensor /= norm
    
    return tensor

def dmrg_sweep(mps, mpo, direction):
    if direction == 'left_to_right':
        for i in range(1, len(mps) - 1):  # start from 1 to avoid boundary issue with mps[0]
            is_edge = (i == 1) or (i == len(mps) - 2)  # Check if the tensor is the second or second-last tensor
            mps[i] = optimize_tensor(mps[i], (mps[i-1], mps[i+1]), mpo[i], is_edge)
    else:
        for i in range(len(mps) - 2, 0, -1):  # go to len(mps) - 2 to avoid boundary issue with mps[-1]
            is_edge = (i == 1) or (i == len(mps) - 2)  # Check if the tensor is the second or second-last tensor
            mps[i] = optimize_tensor(mps[i], (mps[i+1], mps[i-1]), mpo[i], is_edge)
    return mps

# Parameters
N = 100  # Number of sites
J = 1    # Interaction strength
h = 0.5  # External magnetic field

# Initialize MPS and MPO
mps = initialize_mps(N)
mpo = construct_mpo(N, J, h)

# Perform DMRG sweeps
num_sweeps = 1
for _ in range(num_sweeps):
    mps = dmrg_sweep(mps, mpo, 'left_to_right')
    mps = dmrg_sweep(mps, mpo, 'right_to_left')

