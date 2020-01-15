import numpy as np
import theano as t
import theano.tensor as tt
import pymc3 as pm
import Bio.PDB as PDB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rmsd
import time
import subprocess
import os

# there is basically no difference between the mean structures and the rotated and translated means (see plots)
# only a few of the last digits when measuring rsmd between structures differ

import pymc3.distributions.continuous as dist
print("###########################")
print(f"pyMC3 version: {pm.__version__}")
print(f"numpy version: {np.__version__}")
print("###########################")

# variable to change naming of output plots
exists = True
while exists:
    output_name = input("Please type in an output folder name for this run:\n---> ")
    exists = os.path.exists(f'./{output_name}')
    if exists:
        print("This directory already exists. Choose another name.")

# since we can now be sure the directory is not existing, we create it
# -p, --parents	no error if existing, make parent directories as needed -> this won't delete an existing directory
# (even though we checked for that anyway)
process = subprocess.Popen("mkdir -p {}".format(output_name), shell=True)
process.wait()

###########################################
# source: user Danijel at https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python

class Timer:
  def __init__(self):
    self.start = time.time()

  def restart(self):
    self.start = time.time()

  def get_time_hhmmss(self):
    end = time.time()
    m, s = divmod(end - self.start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str
###########################################


def center(m):
    """m : numpy.ndarray representing a n x k matrix, where n is the number of atoms and k is the number of dimensions
     (usually 2D or 3D)
    return: centered matrix m"""
    # calculate center of mass of m
    center_of_mass_m = m.sum(axis=0) / m.shape[0]
    # print(f"center_of_mass_m: {center_of_mass_m}")
    # center m
    centered_m = m - center_of_mass_m
    return centered_m

# To measure the wall clock time of the entire program
total_runtime_timer = Timer()

########### reading in state1 and extracting n random C_alphas

print("Reading in protein structures...")
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("2SDF", "2SDF.pdb")

# a list of list, with the outer list representing the models (= state of the protein, not to be confused with
#  bayesian model) and the inner lists being the c_alpha traces of the loaded protein
model_cas_obs = []
for model in structure:
    cas_obs = []
    for chain in model:
        for residue in chain:
                try:
                    # creates a list of lists containing all residue atom coordinates
                    cas_obs.append(residue['CA'].get_coord())
                except:
                    pass
    model_cas_obs.append(cas_obs)

# number of models (= state conformations) in pdb
num_models = len(model_cas_obs)
# length of C_alpha trace
n = len(model_cas_obs[0])

# assert that all states hve same number of c_alphas
assert(list(map(lambda cas_obs: len(cas_obs), model_cas_obs)) == [n]*num_models)


def vec2matrix(cas_obs):
    """Transform the c_alpha vector into a matrix and center the matrix around the origin,
     which will later serve as the observed mean of the model"""
    # Vertical stack to concatenate list of lists of coordinates into np array
    cas_obs_array = np.vstack(cas_obs)
    cas_obs_array_centered = center(cas_obs_array)
    M_obs = cas_obs_array_centered
    return M_obs

M_obs_state1 = vec2matrix(model_cas_obs[0])
M_obs_state2 = vec2matrix(model_cas_obs[1])

###########


with pm.Model() as inner_model:

### 1. prior over mean M
    x1 = pm.Uniform("x1", 0, 1, shape=n)
    theta = 2 * np.pi * x1

    x2 = pm.Uniform("x2", 0, 1, shape=n)
    phi = tt.arccos(1 - 2 * x2)

    x = tt.sin(phi) * tt.cos(theta)
    y = tt.sin(phi) * tt.sin(theta)
    z = tt.cos(phi)

    # stack the 3 coordinate-column-vectors horizontally to get the n*3 matrix of all C_alpha positions
    # adjust length of unit vectors to 3.8 A (average C_alpha distance)
    cas = pm.Deterministic("cas", 3.8 * tt.stack([x, y, z], axis=1))

    # make a C_alpha-trace out of the C_alpha positions lying on the sphere with radius 3.8
    cas_trace = tt.extra_ops.cumsum(cas, axis=0)

    # Move center of mass of C_alpha trace to origin
    M = cas_trace-tt.mean(cas_trace, axis=0)

### 2. prior over translations T_i
    T1 = pm.Normal('T1', mu=0, sd=100000000, shape=3)
    T2 = pm.Normal('T2', mu=0, sd=100000000, shape=3)

### 3. prior over rotations R_i

    # argument 'i' guarantees that the symbolic variable name will be identical every time this method is called
    # repeating a symbolic variable name in a model will throw an error
    def sample_R(i):
        """sample a unit quaternion and transform it into a rotation matrix"""
        # the first argument states that i will be the name of the rotation made
        ri_vec = pm.Uniform("sample_R" + str(i), 0, 1, shape=3)  # Note shape is 3-Dimensional
        theta1 = 2 * np.pi * ri_vec[1]
        theta2 = 2 * np.pi * ri_vec[2]

        r1 = tt.sqrt(1 - ri_vec[0])
        r2 = tt.sqrt(ri_vec[0])

        # TODO: check q is unit-q
        qw = r2 * tt.cos(theta2)
        qx = r1 * tt.sin(theta1)
        qy = r1 * tt.cos(theta1)
        qz = r2 * tt.sin(theta2)

        # np.eye(3) initializes a 3x3 identity matrix
        # theano.shared(value, ...): Return a SharedVariable Variable, initialized with a copy or reference of value.
        R = t.shared(np.eye(3))

        # filling the rotation matrix
        # Evangelos A. Coutsias, et al "Using quaternions to calculate RMSD" In: Journal of Computational Chemistry 25.15 (2004)

        # Row one
        # tt.sqr = square, tt.sqrt = square-root
        R = tt.set_subtensor(R[0, 0], qw**2 + qx**2 - qy**2 - qz**2)
        R = tt.set_subtensor(R[0, 1], 2*(qx*qy - qw*qz))
        R = tt.set_subtensor(R[0, 2], 2*(qx*qz + qw*qy))
        # R = tt.set_subtensor(R[0, :], [a, b, c])

        # Row two
        R = tt.set_subtensor(R[1, 0], 2*(qx*qy + qw*qz))
        R = tt.set_subtensor(R[1, 1], qw**2 - qx**2 + qy**2 - qz**2)
        R = tt.set_subtensor(R[1, 2], 2*(qy*qz - qw*qx))

        # Row three
        R = tt.set_subtensor(R[2, 0], 2*(qx*qz - qw*qy))
        R = tt.set_subtensor(R[2, 1], 2*(qy*qz + qw*qx))
        R = tt.set_subtensor(R[2, 2], qw**2 - qx**2 - qy**2 + qz**2)
        return R

### 4. putting the model together

    M_T1_model_state1 = M + T1

    # the deterministic symvar doesn't affect the model
    R2 = pm.Deterministic('R2', sample_R(2))
    M_R2_T2_model_state2 = tt.dot(M, R2) + T2

### 5. prior over gaussian noise E_i
    # sample parameters for multivariate distribution

    # standard variance
    sv = pm.HalfNormal("sv", sd=100000000)

    # covariance between atoms
    U = sv * tt.eye(n)  # see Theobalt 3.2 end (U is the first argument)
    # covariance between coordinates
    V = tt.eye(3)  # see Theobalt 3.2 end (V is the second argument)

    # X1
    M_T1_E1_model_state1 = pm.MatrixNormal("E1_state1", mu=M_T1_model_state1, rowcov=U, colcov=V, shape=(n, 3), observed=M_obs_state1)

    # X2
    M_R2_T2_E2_model_state2 = pm.MatrixNormal("E2_state2", mu=M_R2_T2_model_state2, rowcov=U, colcov=V, shape=(n, 3), observed=M_obs_state2)

    # define symbolic variable of estimated M in the model so it is accessible afterwards
    M_model = pm.Deterministic('M_model', M)
    M_T1_E1_model_state1 = pm.Deterministic('M_T1_E1_model_state1', M_T1_E1_model_state1)
    M_R2_T2_E2_model_state2 = pm.Deterministic('M_R2_T2_E2_model_state2', M_R2_T2_E2_model_state2)

my_timer = Timer()

# Bayesian Inference via maximum a posteriori estimation of log-likelihood
print("##### starting MAP estimate ######")
map_estimate = pm.find_MAP(maxeval=10000, model=inner_model)
print("##### MAP estimate done ######")

print(f"Time for MAP estimate: {my_timer.get_time_hhmmss()}")

# is this the way to access symbolic variable from model? (first defined line 171)
M_model = map_estimate["M_model"]

### plot the results in matplotlib and write them into a pdb file

fig = plt.figure(figsize=(18, 16), dpi=80)
fig.add_subplot(111, projection='3d')

M_T1_E1_model_state1 = map_estimate["M_T1_E1_model_state1"]
M_R2_T2_E2_model_state2 = map_estimate["M_R2_T2_E2_model_state2"]

########################################
# Assert that outputs are centered around origin (norm(com(coordinate_matrix)) ~ 0)
def com(m):
   """returns center of mass of matrix m"""
   return m.sum(axis=0) / m.shape[0]

np.testing.assert_almost_equal(np.linalg.norm(com(M_model)), 0, decimal=6)
np.testing.assert_almost_equal(np.linalg.norm(com(M_obs_state1)), 0, decimal=6)
np.testing.assert_almost_equal(np.linalg.norm(com(M_obs_state2)), 0, decimal=6)
np.testing.assert_almost_equal(np.linalg.norm(com(M_T1_E1_model_state1)), 0, decimal=6)
np.testing.assert_almost_equal(np.linalg.norm(com(M_R2_T2_E2_model_state2)), 0, decimal=6)
########################################

# For PDB output
def write_ATOM_line(name, i, x, y, z, aa_type, file_name):
    _ATOM_FORMAT_STRING = "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f %4s%2s%2s\n"
    # use 'a' to append, not 'w' to (over-)write!
    with open(file_name, 'a') as f:
        args=("ATOM  ", i, name, " ", aa_type, "A", i, " ", x, y, z, 0.0, 0.0, "X", " ", " ")
        #f.write(_ATOM_FORMAT_STRING % args)

        #f.write(f"ATOM\t{i}\t{name}\t{aa_type}\tA\t{i}\t{x}\t{y}\t{z}\t0.00\t0.00\tX\t\n")
        f.write(f"ATOM{i:7d} {name}   {aa_type} A{i:4d}{x:12.3f}{y:8.3f}{z:8.3f}  0.00  0.00    X    \n")

# blue graph
x=M_T1_E1_model_state1[:, 0]
y=M_T1_E1_model_state1[:, 1]
z=M_T1_E1_model_state1[:, 2]
plt.plot(x, y, z, label="M_T1_E1_model_state1")
for i in range(len(x)):
    write_ATOM_line("CA",i+1, x[i], y[i], z[i], "ALA", f"{output_name}/2SDF_state1.pdb")

# orange graph
x=M_R2_T2_E2_model_state2[:, 0]
y=M_R2_T2_E2_model_state2[:, 1]
z=M_R2_T2_E2_model_state2[:, 2]
plt.plot(x, y, z, label="M_R2_T2_E2_model_state2")
for i in range(len(x)):
    write_ATOM_line("CA", i+1, x[i], y[i], z[i], "ALA", f"{output_name}/2SDF_state2.pdb")

# green graph
x=M_model[:, 0]
y=M_model[:, 1]
z=M_model[:, 2]
plt.plot(x, y, z, label="M_model")
for i in range(len(x)):
    write_ATOM_line("CA", i+1, x[i], y[i], z[i], "ALA", f"{output_name}/2SDF_mean_structure.pdb")

error_state1 = rmsd.kabsch_rmsd(M_T1_E1_model_state1, M_model)
error_state2 = rmsd.kabsch_rmsd(M_R2_T2_E2_model_state2, M_model)
rmsd_between_states = rmsd.kabsch_rmsd(M_T1_E1_model_state1, M_R2_T2_E2_model_state2)
plt.title(f"Comparison of modeled structures\nrmsd state1 = {error_state1}\nrmsd state2 = {error_state2}\n"
          f"rmsd between states = {rmsd_between_states}")
plt.legend()
plt.savefig(f"{output_name}/comparison_plot.png")
plt.close()

print(f"RMSD of BI model state1:\n{error_state1}")
print(f"RMSD of BI model state2:\n{error_state2}")
print(f"RMSD between BI models:\n{rmsd_between_states}")

print(f"total runtime: {total_runtime_timer.get_time_hhmmss()}")
