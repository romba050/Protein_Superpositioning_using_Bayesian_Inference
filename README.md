Protein Superpositioning using Bayesian Inference
by Basile Rommes Supervisor Thomas Hamelryck April 6, 2018

1

Introduction

One possible Bayesian Interference model for protein superpositioning is the Gaussian perturbation model found in the expectation maximization (EM) approach used in the paper by Theobalt et al. [1] and implemented in the program 'Theseus'. Here, the different protein structures Xi are represented as deviations from a mean structure M . The mean structure M is subject to some Gaussian noise Ei (do to the central limit theorem), specific to each protein Xi , as well as a rotation Ri and a translation Ti . Xi = Ri · (M + Ei ) + Ti If we select M such that it has the same orientation as one of our proteins Xi , we can omit the rotation for one of the structures. When superpositioning only two protein structures, the model takes the form: X1 = (M + E1 ) + T1 X2 = R · (M + E2 ) + T2 Thus X1 and X2 are multidimensional and normally distributed (N3n (µ, )) with the parameters: X 1  N3n (M + T 1, V  U ) X 2  N3n (RM + T 2, V  U ) with covariance matrices V := I3 and U :=  2 In from MN n,p (M, U =  2 In , V = I3 ) (see [1] end of 3.2), where n is the number of C s in Xi (See chapter 3 for a derivation from the matrix normal to the multidimensional normal distribution.) The probability of seeing X1 and X2 under the Gaussian perturbation model is: P (X 1|M, I3   2 I3 ) 1

P (X 2|M, I3  ( 2 ) I3 )

Hereinafter, I shall explain the choice of the priors, which is crucial to the design of the Bayesian Interference model. Moreover, to get from the model to a log-likelihood posterior distribution, we sample from a matrix normal distribution, as described in section 5.

2

Reading in real protein data and describing it according to the model

Our protein of choice is 1ENH, read-in as a pdb file downloaded from RCSB PDB. From the structure, n C coordinates are extracted:
39 40 41 42 43 44 45 46 47 48 49

parser = PDB . PDBParser ( QUIET = True ) structure = parser . get_structure ( " 1 ENH " , " 1 enh . pdb " ) # cas from loaded protein cas_real = [] for model in structure : for chain in model : for residue in chain : try : # creates a list of lists containing all residue atom coordinates cas_real . append ( residue [ ' CA ' ]. get_coord () ) except : pass

50 51 52

The coordinates are put into matrix form and centered around the origin.
56 57 58 59

# Vertical stack to concatenate list of lists of coordinates into np array cas_real_array = np . vstack ( cas_real ) print ( f " cas_real_array :\ n { cas_real_array } " ) c a s _ r e a l _ a r r a y _ c e n t e r e d = center ( cas_real_array )

3

Prior over rotations Ri

Because new random variables have to be sampled from the uniform distribution everytime we define a rotation (see further down how to get from the RV to the rotation), this prior is defined in a python function. Since the random variables have to be saved as symbolic variables with unique names in the pymc3 model, the function sample R takes an integer used in the variable name as argument. Note however, that for this simple case, we only need one rotation, and therefore only call sample R once.

2

112

113

114

115

""" sample a unit quaternion and transform it into a rotation matrix """ # the first argument states that i will be the name of the rotation made ri_vec = pm . Uniform ( 'r ' + str ( i ) , 0 , 1 , testval =0.5 , shape =3) Note shape is 3 - Dimensional theta1 = 2 * np . pi * ri_vec [1]

#

The prior distribution for the rotations Ri was chosen from a uniform distribution of unit quaternions. This method is presented in a book by Xavier Perez-Sala et al.[2]. Let U (.) be the uniform distribution and let q = (w, x, y, z ) be a quaternion of the form w, xi, yj, zk representing a rotation in 3D-space. [The last scalar w represents the angle of rotation and the last three scalars x, y, z represent a point in 3D space, that defines the rotation axis together with the origin?] Xavier Perez-Sala et al. notes that a uniformly distributed element of a complete group (e.g. group of quaternions q = (w, x, y, z )) can be achieved by multiplication of a uniformly distributed element from a subgroup (e.g. the group of planar rotations around the Z-axis: q = (c, 0, 0, s)) with a uniformly distributed coset (e.g. rotations of the Z-axis: q = (w, x, y, 0)). U ([c, 0, 0, s])U ([w, x, y, 0]) = U ([cw, cx + sy, -sx + cy, sw]) We get the quaternion by sampling the parameters 1 , 2 , r1 and r2 and plugin them into this expression: q = (w, x, y, z ) = (r2 cos 2 , r1 sin 1 , r1 cos 1 , r2 sin 2 ) Let's have a look at how this is done in theano:
120 121 122

qx = r1 * tt . sin ( theta1 ) qy = r1 * tt . cos ( theta1 ) qz = r2 * tt . sin ( theta2 )

Three random variables X0 , X1 , X2 are sampled in the code (bayesian inference.py):
111

ri_vec = pm . Uniform ( 'r ' + str ( i ) , 0 , 1 , testval =0.5 , shape =3) Note shape is 3 - Dimensional

#

Where X1 and X2 define the  angles:
111 112

theta1 = 2 * np . pi * ri_vec [1] theta2 = 2 * np . pi * ri_vec [2]

While X0 defines the two radii r1 and r2 :
116

r2 = tt . sqrt ( ri_vec [0])

3

The relation between quaternion q = (w, x, y, z ) and a 3 × 3 rotation matrix is (see Coutsias et al 2004 [3])  2  w + x2 - y 2 - z 2 2(xy - wz ) 2(xz + wy ) w 2 - x2 + y 2 - z 2 2(yz - wx)  (w, x, y, z ) =  2(xy + wz ) 2(xz - wy ) 2(yz + wx) w 2 - x2 - y 2 + z 2
126 127 128 129

R = t . shared ( np . eye (3) ) # filling the rotation matrix # Evangelos A . Coutsias , et al " Using quaternions to calculate RMSD " In : Journal of Computational Chemistry 25.15 (2004) # # R R R # # R R R # R R R Row one tt . sqr = square , tt . sqrt = square - root = tt . set_subtensor ( R [0 , 0] , qw **2 + qx **2 - qy **2 - qz **2) = tt . set_subtensor ( R [0 , 1] , 2*( qx * qy - qw * qz ) ) = tt . set_subtensor ( R [0 , 2] , 2*( qx * qz + qw * qy ) ) R = tt . set_subtensor ( R [0 , :] , [a , b , c ]) Row two = tt . set_subtensor ( R [1 , 0] , 2*( qx * qy + qw * qz ) ) = tt . set_subtensor ( R [1 , 1] , qw **2 - qx **2 + qy **2 - qz **2) = tt . set_subtensor ( R [1 , 2] , 2*( qy * qz - qw * qx ) ) Row three = tt . set_subtensor ( R [2 , 0] , 2*( qx * qz - qw * qy ) ) = tt . set_subtensor ( R [2 , 1] , 2*( qy * qz + qw * qx ) ) = tt . set_subtensor ( R [2 , 2] , qw **2 - qx **2 - qy **2 + qz **2)

130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146

This rotation is an uninformative prior, i.e. it is chosen for the sole reason of weighing all rotations equally likely (uniform distribution), without introducing any bias.

4

Prior mean structure M

The mean structure is modelled as a chain of C -atoms. This protein backbone can be conceptionalized as a sequence of vectors, representing the direct transitions from one C position to the next. We can represent those vectors via polar-coordinates (r, , ) where r is the vector's length, and since all vectors are of same length, r is constant and can be interpreted as the radius of a sphere.  is the azimuthal angle and  the polar angle on this sphere (see the online article by Cory Simon[4]). The relationship between cartesian (x, y, z ) coordinates and polar-coordinates (r, , ) is: x = r · cos()sin() 4

y = r · sin()sin() z = r · cos() In theano:
83 84 85

x = tt . sin ( phi ) * tt . cos ( theta ) y = tt . sin ( phi ) * tt . sin ( theta ) z = tt . cos ( phi )

As Cory Simon explains, due to the way a sphere's surface area in polar-coordinates is calculated, sampling  and  from a uniform distribution doesn't result in uniformly distributed points on a sphere. I.e. vectors from origin to sphere surface that are sampled by this method will not be uniformly distributed. The first approach gives uniform points on a [0, 2 [×[0,  [ square (with  and  the respective coordinates) but achieving uniform distribution on a sphere requires a different approach. The detailed derivation can be read in [4], but it boils down to the formula: P{F -1 (U )  x} = F () Where U is the uniform distribution and F is the cumulative distribution function. This makes the algorithm for sampling the distribution f () via inverse transformation sampling: · Sample u from U (0, 1).
80

x2 = pm . Uniform ( " x2 " , 0 , 1 , shape = n )

· Compute  = F -1 (u), in the 2-sphere case: F -1 (u) = arccos(1 - 2u).
81

phi = tt . arccos (1 - 2 * x2 )

·  can be treated as a random number drawn from the distribution f () Meanwhile  is calculated straightforward via 2v where v is uniformly sampled.
77 78

x1 = pm . Uniform ( " x1 " , 0 , 1 , shape = n ) theta = 2 * np . pi * x1

For this simple prior, we are going to assume equal distances of 3.8° A between neighbouring C 's.
89

cas = pm . Deterministic ( " cas " , 3.8* tt . stack ([ x , y , z ] , axis =1) )

In order to get the C trace - a connected chain of the C atoms - we need to sum over all unit vectors. The resulting trace is then centered to the origin by substracting it's center of mass.
92

cas_trace = tt . extra_ops . cumsum ( cas , axis =0) M = cas_trace - tt . mean ( cas_trace , axis =0)

96

5

5

Gaussian Prior

The protein structures consisting out of p atoms are represented in our model as n×3 matrices Xi (just as in Theobalt et al. [1]). To model Gaussian noise around the mean, we sample from the matrix normal distribution. A multivariate normal distribution Nk (µ, ) gives the probability density of k independent gaussian distributed random variables X1 , .., Xk with mean vector µ and covariance matrix . The matrix normal distribution MN n,p (M, U, V ) is a generalization of the multivariate normal distribution to matrix-valued random variables, where: · M is the n × p mean matrix · U is the n × n covariance among rows matrix, in our case that is the covariance among atom-positions · V is the p × p covariance among columns matrix, in our case a 3 × 3 matrix with covariance among x-, y- and z-coordinates of atoms Since Version 3.3 pyMC3 implements the matrix normal via the function MatrixNormal.
224

225

X1 = pm . MatrixNormal ( " M_E1_T1 " , mu = M_T1 , rowcov =U , colcov =V , shape =( n , 3) , observed = c a s _ r e a l _ a r r a y _ c e n t e r e d ) X2 = pm . MatrixNormal ( " M_E2_T2 " , mu = M_R2_T2 , rowcov =U , colcov =V , shape =( n , 3) , observed = c a s _ r e a l _ a r r a y _ c e n t e r e d )

6

Calculate the prior over translations Ti

The translations are vectors in R3 that are drawn from a normal distribution. To have equal probability of all directions a mean of µ = 0 is chosen. To increase the consistency of the model a small standard deviation of sigma = 0.1 was chosen for now.
100

T2 = pm . Normal ( ' T2 ' , mu =0 , sd =0.1 , shape =3)

This work is based on a project done by William Paul Bullock under the supervision of Thomas Hamelryck [5].

7

Maximum a posteriori estimate

The MAP estimation on our model is called, With a cap of the number iterations set to 10000.
287

map_estimate = pm . find_MAP ( maxeval =10000 , model = inner_model )

The model terminates after 790 iterations.

6

8
8.1

Results
1ENH

Comparison: standard deviation of 0.1 against 1. Albeit changing the standard deviation, the RMSD remains identical up to the 6th decimal place. The model with sd=1 terminated earlier, after 540 instead of 790 iterations.

Figure 1: C trace sampled from protein 1ENH (blue) and the transformed mean structure calculated based on the model (orange). A standard deviation of 0.1 was used for sampling of translation and rotation parameters.

7

Figure 2: C trace sampled from protein 1ENH (blue) and the transformed mean structure calculated based on the model (orange). A standard deviation of 1 was used for sampling of translation and rotation parameters. PyMC3's function traceplot offers a way to visualize and summarize the sample output. The left column shows a smoothed histogram of the marginal posterior distribution for each stochastic random variable used in the model. Marginal means that the probabilities are without reference to the value of the other random variables, i.e. non-conditional probabilities. The right column contains the samples of the Markov Chain plotted in sequential order. This trace was drawn using 10 samples via pymc3.sample. Below are the trace plots for a standard deviation of 0.1 and 1 respectively.

8

Figure 3: sd=0.1 9

Figure 4: sd=1 10

8.2

2SDF:

The pdb file for 2SDF consists of 30 states. Each of them feature a small -helix and three  -sheets that stay almost completely rigid in all 30 conformational states, as do the loops in-between them. The polypeptide chains at both ends of the protein however, feature a high degree of conformational variability.

Figure 5: "ML and LS superposition of simulated protein structure data. (A) The true superposition, generated from a known mean structure with known covariance matrices, before arbitrary translations and rotations have been applied. In generating the simulated data, the set of alpha carbon atoms from model 1 of Protein Data Bank entry 2SDF (www.pdb.org) was used as the mean form (67 atoms/landmarks, squared radius of gyration = 152 2). The non-diagonal 67 67 landmark covariance matrix was based on values calculated from the superposition given in 2SDF, with variances ranging from 0.01 to 80 2 and correlations ranging from 0 to 0.99 (see Data Sets 1 and 2). The known dimensional covariance matrix had eigenvalues of 0.16667, 0.33333, and 0.5 corresponding to the x, y, and z axes of 2SDF model 1, respectively. (B) An ordinary LS superposition of the simulated data. (C) A ML superposition of the simulated data, assuming a diagonal landmark covariance matrix  (i.e., no correlations),  = 1, and inverse gamma distributed variances" 11

Figure 6: State 1 of 2SDF

References
[1] Douglas L. Theobald and Phillip A. Steindel. "Optimal simultaneous superpositioning of multiple structures with missing data". In: Bioinformatics 28.15 (2012), pp. 1972­ 1979. issn: 13674803. doi: 10.1093/bioinformatics/bts243. [2] Xavier Perez-Sala et al. Uniform Sampling of Rotations for Discrete and Continuous Learning of 2D Shape Models. IGI Global, Jan. 1970. [3] Evangelos A. Coutsias, Chaok Seok, and Ken A. Dill. "Using quaternions to calculate RMSD". In: Journal of Computational Chemistry 25.15 (2004), pp. 1849­1857. issn: 01928651. doi: 10.1002/jcc.20110. arXiv: arXiv:1011.1669v3. [4] Article by Cory Simon. url: http://corysimon.github.io/articles/uniformdistnon-sphere/ (visited on Feb. 23, 2018). [5] William Paul Bullock. Probabilistic Programming for Protein Superpositioning.

12

