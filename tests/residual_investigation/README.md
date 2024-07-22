# Purpose and Methodology

The following tests are developed to ensure that the computed residuals, $r_{I}$ and $r_{II}$, using Python routines, give results that are close to zero. In `UMAT` subroutines (With UMAT, we mean physics engine subroutines which are written in Fortran), we take a tolerance of 1e-11 as the target. We expect that using Python routines, we can achiece the same tolerance when we use the converged values for $\gamma$ and slip resistance $s$. To do this check, we first do a simulation using UMAT with an arbitrary deformation gradient and orientation. We collect the converged values of $\gamma$ and slip resistance $s$. We then calculate both residuals, $r_I$ and $r_{II}$, for each pair of these values at $t_n$ and $t_{n+1}$, where $n$ covers the entire range of the simulation (usually a few hundred points).

With the collected datapoints from `UMAT`, we also perform two additional tests:

- For each pair of converged quantities, $\gamma_n, s_n$ and $\gamma_{n+1}, s_{n+1}$ and plastic deformation gradient ${\bf F}_{{\rm p},n}$, we calculate the Cauchy stress and compare it with the one calculated using `UMAT`.
- We calculate the __predictive__ value for $r_{II}$, assuming $\gamma_{n+1} = \gamma_n$ and $s_{n+1} = s_n$. It will be useful to know of the amplitude of $r_{II}$ if later we want to implement a scheme that resembles predictor-corrector scheme. We refer to this value as $(r_{II})_p$

# How to run the tests

Simply run `make` to first collect the residuals and errors, that is $r_I$, $r_{II}$, $(r_{II})p$ and $\|\sigma_{\rm UMAT} - \sigma\|/\|\sigma_{\rm UMAT}\|$, and direct them to `out.csv`. Next, loop over the collected results and compare them against the residual `1e-11` and print relevant information in case values are found bigger than the residual. We do not include $(r_{II})_p$ in this comparison.

# Results

All three computed residuals and relative errors are below the threshold, with a few exceptions for $r_{II}$. The values that violate the threshold are of the order `1e-10`, which is in good agreement with the results from `UMAT`.
