# Final Fig. S2 — Frenkel single-site benchmark

## Plot-only rebuild

From the project root, regenerate the publication figure from the cached HEOM,
exact-cumulant, BA, and SCBA arrays with:

```bash
python3 Figures/SI/FigS2/plot_FigS2.py
```

The command writes `Figures/SI/FigS2/FigS2.pdf` (and a PNG preview) without
running either calculation driver.

The revision-28 renderer imports `Figures/common/si_plot_style.py`: exact is
black solid, single-shot BA is neutral-gray dashed, SCBA is red dashed, and
HEOM is the shared light-blue fill.  Typography, inward ticks, unboxed legend,
and panel letters in a reserved band above every axes use the same SI-wide
vector-export settings. Wider column and row gutters and three interior
energy ticks prevent neighboring endpoint labels from colliding.

## Model and conventions

The FigS2 benchmark is the independent-boson Frenkel-transition model

\[
H=\epsilon_x x^\dagger x+\sum_q\omega_q b_q^\dagger b_q
  +x^\dagger x\sum_q c_q(b_q+b_q^\dagger).
\]

The bath has the Drude spectral density

\[
J(\omega)=\frac{2\lambda\gamma\omega}{\omega^2+\gamma^2},
\]

at 300 K. The twelve cases use
\(\lambda=15,30,45,60\) meV and
\(\gamma=12.4,24.8,37.2\) meV. The relaxed transition is
\(E_{00}=2.0\) eV, the untransformed vertical transition is
\(\epsilon_x=E_{00}+\lambda\), and the retarded electronic half-width is
0.1 meV. No additional polaron shift is introduced.

The excitation is a restricted optical transition in the zero-excitation
linear-response sector. Its retarded Green function is

\[
G_x^R(\omega)=
\frac{1}{\omega-\epsilon_x+i\eta-\Sigma_x^R(\omega)}
\]

and is normalized as \(G_x^R(\omega)\sim1/\omega\).

## Production single-shot BA and G-only SCBA

The calculation iterates the continuous finite-temperature
convolution

\[
\Sigma_x^R(\omega)=
\frac{1}{\pi}\int_0^\infty d\nu\,J(\nu)\left[
(\overline{n}(\nu)+1)G_x^R(\omega-\nu)
+\overline{n}(\nu)G_x^R(\omega+\nu)
\right].
\]

Here \(\overline{n}(\omega)=(e^{\beta\omega}-1)^{-1}\).

The emission and absorption terms are evaluated as separate zero-padded
linear FFT convolutions on a uniform \(-1\)--\(4\) eV electronic grid with
0.05 meV spacing.  Endpoint half weights are included, periodic wraparound
is forbidden, and the omitted high-frequency Drude emission tail is restored
from the analytic bare-propagator asymptotic expansion.  Thus no finite set
of physical bath modes is introduced in the BA or SCBA curves.

The single-shot Born approximation evaluates this kernel once with the
noninteracting propagator \(G_{x,0}^R\), then inserts the resulting
\(\Sigma_{\mathrm{BA}}^R\) into the final Dyson equation. Its saved mode
name is `single_shot_ba`.

In the electronic comparison figure, the exact-cumulant peak sets the
vertical scale. HEOM, single-shot BA, and SCBA are not independently
peak-normalized; each is rescaled by one constant so that all four curves
in a panel have the same integrated area.

The thermal Drude bath remains fixed, so this is G-only SCBA.

## Tangent-grid alternative

For regression and nonuniform-grid applications, the code retains the
alternative tangent discretization

\[
\omega_q=\gamma\tan\!\left[
\frac{\pi}{2}\left(1-\frac{q}{N_b+1}\right)\right],
\qquad
c_q=\sqrt{\frac{\lambda\omega_q}{N_b+1}},
\]

for which
\[
\sum_qc_q^2f(\omega_q)\longrightarrow
\int_0^\infty\frac{d\nu}{\pi}J(\nu)f(\nu).
\]
It is not used for the published final Fig. S2 curves.

## Coupled G-D SCBA

Define the collective bath coordinate

\[
X=\sum_qc_q(b_q+b_q^\dagger).
\]

Its bare retarded propagator and spectral function are

\[
D_{X,0}^R(\nu)=
-\frac{2\lambda\gamma}{\gamma-i\nu},
\qquad
B_{X,0}(\nu)=-2\operatorname{Im}D_{X,0}^R(\nu)=2J(\nu).
\]

The coupled mode iterates

\[
G_x^R=[(G_{x,0}^R)^{-1}-\Sigma_x^R]^{-1},
\qquad
D_X^R=[(D_{X,0}^R)^{-1}-\Pi_{nn}^R]^{-1}.
\]

For a dressed equilibrium bath line, the electronic skeleton self-energy
is evaluated using the full convolution

\[
\Sigma_x^R(\omega)=\int_0^\infty\frac{d\nu}{2\pi}B_X(\nu)
\left[
(\overline{n}(\nu)+1)G_x^R(\omega-\nu)
+\overline{n}(\nu)G_x^R(\omega+\nu)
\right].
\]

The direct quadrature uses nonperiodic interpolation, so there is no FFT
wraparound.

Although the phonons are thermally occupied, the electronic transition
is not. Thus \(G_x^<=0\). Every connected Holstein density bubble
contains an occupied electronic lesser line:

\[
\Pi_{nn}^<(\nu)\propto-i\int\frac{d\omega}{2\pi}
G_x^<(\omega)G_x^>(\omega-\nu)=0,
\]

\[
\Pi_{nn}^>(\nu)\propto-i\int\frac{d\omega}{2\pi}
G_x^>(\omega)G_x^<(\omega-\nu)=0.
\]

Therefore \(\Pi_{nn}^R=0\), \(D_X=D_{X,0}\), and the G-only and G-D
spectra overlap to numerical precision. A nonzero phonon renormalization
would require a separately defined finite-density or pumped electronic
state. The Hartree displacement also vanishes in the vacuum.
