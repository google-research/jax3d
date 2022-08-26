# Generative Neural Fields

This directory contains code for projects relating to generative neural fields.

# Coding Conventions

Throughout the code we adhere to the following characters for all `einops` and
`etils.FloatArray` dimensionality annotations.

```
B - number of examples in batch
N - general purpose variable for use when nothing else appropriate.
    eg. If a dimension might be B or S depending on who calls it, then use N.
I - number of identities
V - number of views
K - number of images (I x V)
H - height
W - width
P - pixel (Note a pixel may be sampled by more than one ray)
R - number of rays
S - number of samples (usually per ray, but possibly independent)
L - number of lights or light directions
Z - dimension of latent code
    - and when necessary:
    - Z_i identity latent code
    - Z_v per-identity, per-view latent code
    - Z_e expression conditioning vector (eg. BFM expression coefficients)
    - Z_l1, Z_l2,... illumination parameters - specific to each type of light.
D - degrees of freedom. Various models flatten their human understandable
    parameters into a "degrees of freedom" vector of this dimension.
```