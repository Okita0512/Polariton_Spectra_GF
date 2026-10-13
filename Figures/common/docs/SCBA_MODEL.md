# Frequency-coupled G-only SCBA used by the active manuscript figures

The retained solver roots support SI Fig. S9 (`FigS9`), main Fig. 2, the
shared detuning cache used by main Fig. 3 and SI Fig. S10 (`FigS10`), and
SI Fig. S12 (`FigS12`).

Each legacy case is a single-excitation Holstein-Tavis-Cummings model with
one local Drude bath per molecule.  The untransformed vertical exciton energy
is \(E_x=E_{00}+\lambda\), and

\[
J(\nu)=\frac{2\lambda\gamma\nu}{\nu^2+\gamma^2}.
\]

The retarded bright and dark exciton Green functions are evaluated in the
cavity/bright basis:

\[
G_B^R(\omega)=
\frac{\omega-\omega_c+i(\kappa+\eta)}
 {[\omega-\omega_c+i(\kappa+\eta)]
  [\omega-E_x+i\eta-\Sigma^R(\omega)]-\Omega_R^2},
\]

\[
G_D^R(\omega)=
\frac{1}{\omega-E_x+i\eta-\Sigma^R(\omega)}.
\]

Because a local phonon vertex addresses one molecule, the Green function
inside the Born skeleton is

\[
G_{\rm loc}^R(\omega)=
\frac{G_B^R(\omega)+(N-1)G_D^R(\omega)}{N}.
\]

The G-only SCBA equation is iterated on the complete frequency grid:

\[
\Sigma^R(\omega)=\sum_q c_q^2\left[
(\overline{n}(\omega_q)+1)G_{\rm loc}^R(\omega-\omega_q)
+\overline{n}(\omega_q)G_{\rm loc}^R(\omega+\omega_q)\right].
\]

For production this sum is evaluated in its continuous form,

\[
\Sigma^R(\omega)=\frac{1}{\pi}\int_0^\infty d\nu\,J(\nu)
\left[(\overline{n}(\nu)+1)G_{\rm loc}^R(\omega-\nu)
+\overline{n}(\nu)G_{\rm loc}^R(\omega+\nu)\right],
\]

where \(\overline{n}(\omega)=(e^{\beta\omega}-1)^{-1}\) is the
Bose--Einstein distribution function.

The two one-sided terms are computed as zero-padded linear FFT convolutions,
with trapezoidal endpoint weights and an analytic asymptotic correction for
the omitted high-frequency Drude emission tail.  There is no circular
wraparound.  Thus a shifted propagator uses the self-energy at the shifted
frequency, and there is no independent scalar Dyson loop at each output
point.

The phonon propagator remains its equilibrium bare value.  Linear absorption
uses the zero-exciton reference state, for which the connected Holstein
density polarization is zero; a phonon self-consistency loop would therefore
return \(D=D_0\).

The production convolution grid spans -1 to 4 eV.  Its spacing is 0.05 meV
when the retarded half-width is 0.1 meV and 0.01 meV when the half-width is
0.01 meV.  The single-shot BA provides the initial condition and the maximum
SCBA residual tolerance is \(10^{-8}E_{\rm h}=2.72\times10^{-7}\) eV.  The converged
self-energy is interpolated onto each legacy output grid before spectra and
linewidths are evaluated.

The previous 1000-mode tangent discretization remains available through
`SCBA_BATH_METHOD=tangent` or `--bath-method tangent`.  It is retained as an
alternative quadrature and validation route, not as the production method.
