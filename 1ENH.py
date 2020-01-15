import numpy as np
import theano as t
import theano.tensor as tt
import pymc3 as pm
from theano.tensor.shared_randomstreams import RandomStreams
import Bio.PDB as PDB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# RMSD as measure of difference between our estimation and the true structure
# applies no further translation or rotation
import rmsd as rmsd
import time
import subprocess
import os

import pymc3.distributions.continuous as dist
print("###########################")
print(f"pyMC3 version: {pm.__version__}")
print(f"numpy version: {np.__version__}")
print("###########################")

SEED = 234

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
    """
    m : numpy.ndarray representing a n x k matrix, where n is the number of atoms and k is the number of dimensions
     (usually 2D or 3D)
    return: centered matrix m"""
    # calculate center of mass of x
    center_of_mass_m = m.sum(axis=0) / m.shape[0]
    # print(f"center_of_mass_m: {center_of_mass_m}")
    # center m
    centered_m = m - center_of_mass_m
    return centered_m

# To measure the wall clock time of the entire program
total_runtime_timer = Timer()

########### reading in 1ENH and extracting n random C_alphas

parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("1ENH", "1enh.pdb")

# cas from loaded protein
cas_real = []
for model in structure:
    for chain in model:
        for residue in chain:
                try:
                    # creates a list of lists containing all residue atom coordinates
                    cas_real.append(residue['CA'].get_coord())
                except:
                    pass

# length of C_alpha trace
n = len(cas_real)

# sample n successive atoms from protein, starting at a random position:
##### rand = np.random.randint(len(cas_real)-n)
##### print(f"start position of real C_alpha trace within loaded protein: {rand}")
##### cas_real = cas_real[rand: rand+n] #####################################
# Vertical stack to concatenate list of lists of coordinates into np array
cas_real_array = np.vstack(cas_real)
cas_real_array_centered = center(cas_real_array)
M_obs = cas_real_array_centered
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
    cas = pm.Deterministic("cas", 3.8*tt.stack([x, y, z], axis=1))

    # make a C_alpha-trace out of the C_alpha positions lying on the sphere with radius 3.8
    cas_trace = tt.extra_ops.cumsum(cas, axis=0)

    # Move center of mass of C_alpha trace to origin
    M = cas_trace-tt.mean(cas_trace, axis=0)

### 2. prior over translations T_i
    testval = np.random.normal()
    T1 = pm.Normal('T1', mu=0, sd=1, shape=3)
    testval = np.random.normal()
    T2 = pm.Normal('T2', mu=0, sd=1, shape=3)

### 3. prior over rotations R_i

    # argument i garantees that the symbolic variable name will be identical everytime this method is called
    # repeating a symbolic variable name in a model will throw an error
    def sample_R(i):
        """sample a unit quaternion and transform it into a rotation matrix"""
        # the first argument states that i will be the name of the rotation made
        ri_vec = pm.Uniform("sample_R" + str(i), 0, 1, shape=3)  # Note shape is 3-Dimensional
        theta1 = 2 * np.pi * ri_vec[1]
        theta2 = 2 * np.pi * ri_vec[2]

        r1 = tt.sqrt(1 - ri_vec[0])
        r2 = tt.sqrt(ri_vec[0])

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

    M_T1 = M + T1

    # the deterministic symvar is ignored
    R2 = pm.Deterministic('R2', sample_R(2))
    M_R2_T2 = tt.dot(M, R2) + T2


    # define symbolic variable of estimated M in the model so it is accessible afterwards
    M_estimation = pm.Deterministic('M_estimation', M)

    ### 5. prior over gaussian noise E_i
    # sample parameters for multivariate distribution
    sv = pm.HalfNormal("sv", sd=1)

    U = sv * tt.eye(n)  # see Theobalt 3.2 end (U is the first argument)
    V = tt.eye(3)  # see Theobalt 3.2 end (V is the second argument)

    # X1
    M_E1_T1 = pm.MatrixNormal("M_E1_T1", mu=M_T1, rowcov=U, colcov=V, shape=(n, 3), observed=M_obs)

    # X2
    M_E2_T2 = pm.MatrixNormal("M_E2_T2", mu=M_R2_T2, rowcov=U, colcov=V, shape=(n, 3), observed=M_obs)

with pm.Model() as outer_model:
    with inner_model:
        # draw one sample, it returns a datastructure named trace
        print("###### Sampler is called ######")
        my_timer = Timer()
        trace = pm.sample(1, random_seed=SEED)
        time_hhmmss = my_timer.get_time_hhmmss()
        print(f"Time for sampler: {time_hhmmss}")
        print("###### Sampler is done ######")
    print(trace)
    print("x1")
    # "The first dimension of the array is the sampling index and the later dimensions match the shape of the variable.
    print(trace["x1"][0]) # array of length nm RV used for sampling angles
    print("T1")
    print(trace["T1"][0])
    print("###############")
    print("T1 shape")
    print(trace["T1"].shape)
    print("T1[0].shape")
    print(trace["T1"][0].shape)
    print("###############")
    print("T2")
    print(trace["T2"][0])
    print("R2")
    print(trace["R2"][0])
    print("sample_R2")
    print(trace["sample_R2"][0])
    print("M_estimation")
    print(trace["M_estimation"][0])
    print("sv")
    print(trace["sv"][0])
    print("cas")
    print(trace["cas"][0]) # this thing is a matrix!
    print(f"trace['cas'][0].shape: {trace['cas'][0].shape}")
    print(f"np.arange(10).shape: {np.arange(10).shape}")
    print(f"trace['x1'].T.shape: {trace['x1'].T.shape}")
    plt.savefig(f"{output_name}/x1_trace.png")
    plt.close()

    # axis = 1 to turn nx3 matrix into n dim vector of norms
    np.testing.assert_almost_equal(np.linalg.norm(trace["cas"][0], axis = 1), 3.8*np.ones(shape=n))

# Plot the trace for the MCMC sampling:
# allow for some iterations to pass before plotting so parameters can settle in (see burn-in)
burnin = 0
trace = trace[burnin:]
#pm.summary(trace)


pm.traceplot(trace)
plt.savefig(f"{output_name}/MCMC_trace.png")

my_timer = Timer()

# Bayesian Inference via maximum a posteriori estimation of log-likelihood
print("##### MAP estimate ######")
map_estimate = pm.find_MAP(maxeval=10000, model=inner_model)
print("##### MAP estimate done ######")

print(f"Time for MAP estimate: {my_timer.get_time_hhmmss()}")

# is this the way to access symbolic variable from model? (first defined line 171)
M_estimation = map_estimate["M_estimation"]

### plot results

fig = plt.figure(figsize=(18, 16), dpi=80)
ax = fig.add_subplot(111, projection='3d')

# blue graph
x=M_obs[:, 0]
y=M_obs[:, 1]
z=M_obs[:, 2]
plt.plot(x, y, z)

# orange graph
x2=M_estimation[:, 0]
y2=M_estimation[:, 1]
z2=M_estimation[:, 2]
plt.plot(x2, y2, z2)

error = rmsd.kabsch_rmsd(M_obs, M_estimation)
plt.title(f"rmsd = {error}")
plt.savefig(f"{output_name}/comparative_plot.png")

print(f"total runtime: {total_runtime_timer.get_time_hhmmss()}")
