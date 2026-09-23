# Nondimensionalization (physical ↔ code units)

Everything in SPEcQK is dimensionless. This page shows how the code's variables
relate to physical (SI) quantities, so you can convert your problem into the
inputs of [`run_1d`](@ref) and convert the output moments back.

The anchor is the Maxwellian used everywhere in the code
([`maxwell_boltzmann!`](@ref)):

```math
f(\mathbf v) \;\propto\; \exp\!\left(-\frac{|\mathbf v-\mathbf u|^2}{T}\right),
```

i.e. the temperature `T` sits in the exponent *without* the usual factor
``m/2k_B``. Matching this to the physical Maxwellian fixes the entire scaling.

## Reference quantities

Pick three independent physical references: the particle mass ``m``, a reference
number density ``n_\mathrm{ref}`` and a reference temperature ``T_\mathrm{ref}``.
A reference length ``L_\mathrm{ref}`` (typically the physical size of the spatial
domain) closes the system. Everything else is derived:

| Quantity        | Reference scale                                            | Meaning                                   |
|-----------------|------------------------------------------------------------|-------------------------------------------|
| velocity        | ``V_\mathrm{ref}=\sqrt{\dfrac{2k_B T_\mathrm{ref}}{m}}``    | most-probable thermal speed at ``T_\mathrm{ref}`` |
| length          | ``L_\mathrm{ref}``                                          | domain size (your choice)                 |
| time            | ``t_\mathrm{ref}=L_\mathrm{ref}/V_\mathrm{ref}``           | flow-through time                         |
| number density  | ``n_\mathrm{ref}``                                          |                                           |
| temperature     | ``T_\mathrm{ref}``                                          |                                           |
| VDF             | ``f_\mathrm{ref}=n_\mathrm{ref}/V_\mathrm{ref}^3``         | so that ``\int f\,d^3v`` is a density     |
| pressure        | ``p_\mathrm{ref}=n_\mathrm{ref}k_B T_\mathrm{ref}``        |                                           |
| dyn. viscosity  | ``\mu_\mathrm{ref}=p_\mathrm{ref}\,t_\mathrm{ref}=\dfrac{n_\mathrm{ref}k_B T_\mathrm{ref}L_\mathrm{ref}}{V_\mathrm{ref}}`` | (see the BGK section) |

Here ``k_B`` is the Boltzmann constant. This is the ``V_\mathrm{ref}=\sqrt{2k_BT/m}``
convention: the thermal speed scale is ``V_\mathrm{ref}``, so at ``T=1`` bulk and
thermal velocities are ``\mathcal O(1)``.

## Dimensionless variables

Writing physical quantities with a hat, the code variables are:

```math
\mathbf v=\frac{\hat{\mathbf c}}{V_\mathrm{ref}},\quad
\mathbf u=\frac{\hat{\mathbf U}}{V_\mathrm{ref}},\quad
x=\frac{\hat x}{L_\mathrm{ref}},\quad
t=\frac{\hat t}{t_\mathrm{ref}},\quad
n=\frac{\hat n}{n_\mathrm{ref}},\quad
T=\frac{\hat T}{T_\mathrm{ref}},\quad
f=\frac{\hat f}{f_\mathrm{ref}},\quad
p=\frac{\hat p}{p_\mathrm{ref}} = nT .
```

The temperature relation, written out, is

```math
T=\frac{\hat T}{T_\mathrm{ref}}=\frac{2k_B\hat T}{m\,V_\mathrm{ref}^2}.
```

### Why this reproduces `exp(-v²/T)`

The physical Maxwellian is
``\hat f(\hat{\mathbf c})=\hat n\left(\frac{m}{2\pi k_B\hat T}\right)^{3/2}
\exp\!\big(-\frac{m|\hat{\mathbf c}-\hat{\mathbf U}|^2}{2k_B\hat T}\big)``.
Substituting ``\hat{\mathbf c}=V_\mathrm{ref}\mathbf v`` and dividing by
``f_\mathrm{ref}`` gives exactly

```math
f(\mathbf v)=n\,(\pi T)^{-3/2}\exp\!\left(-\frac{|\mathbf v-\mathbf u|^2}{T}\right),
```

because ``\big(\frac{m}{2\pi k_B\hat T}\big)^{3/2}V_\mathrm{ref}^3=(\pi T)^{-3/2}``.
The exponent argument ``m/2k_B\hat T`` becomes ``1/T`` — that is the whole point
of choosing ``V_\mathrm{ref}=\sqrt{2k_BT_\mathrm{ref}/m}``.

Consequences (all verified numerically against `maxwell_boltzmann!`):

* per-direction velocity **variance** in code units is ``T/2``
  (``\langle(v_x-u_x)^2\rangle=T/2``); the physical standard deviation is
  ``V_\mathrm{ref}\sqrt{T/2}=\sqrt{k_B\hat T/m}`` as expected;
* the most-probable speed at temperature ``T`` is ``\sqrt{T}`` in code units,
  i.e. ``V_\mathrm{ref}\sqrt{T}=\sqrt{2k_B\hat T/m}`` physically.

## Moments

The code moment with powers ``(a,b,c)`` is the **raw** (not density-normalized)
integral

```math
M_{abc}=\int v_x^{a}v_y^{b}v_z^{c}\,f(\mathbf v)\,d^3v .
```

Its physical counterpart follows from ``\hat f\,d^3\hat c=n_\mathrm{ref}f\,d^3v``
and ``\hat c_i=V_\mathrm{ref}v_i``:

```math
\mathcal M_{abc}=\int \hat c_x^{a}\hat c_y^{b}\hat c_z^{c}\,\hat f\,d^3\hat c
   \;=\; n_\mathrm{ref}\,V_\mathrm{ref}^{\,a+b+c}\;M_{abc}.
```

Primitive variables from the code moments (this is exactly what the reconstruction
and [`find_MB_solution_T_and_v!`](@ref) use):

```math
n=M_{000},\qquad
\mathbf u=\frac{1}{n}\,(M_{100},M_{010},M_{001}),
```
```math
T=\frac{2}{3}\!\left(\frac{M_{200}+M_{020}+M_{002}}{n}-|\mathbf u|^2\right),
\qquad p=nT .
```

<!-- The factor ``2/3`` is the counterpart of the missing ``m/2k_B`` in the exponent:
``\frac{2}{3}\langle|\mathbf c-\mathbf u|^2\rangle = T`` because each direction
contributes a variance of ``T/2``. -->

Physical pressure is ``\hat p=p_\mathrm{ref}\,nT=\hat n k_B\hat T``.

### Hydrodynamic formulas: the code gas constant is ``R=1/2``

!!! warning "`p = nT` is *not* the pressure that goes into Euler/NS formulas"
    Use ``p_\text{mech}=nT/2``, not ``nT``, in the sound speed, the
    Rankine–Hugoniot relations, any Riemann solver, and any Mach number.

Both statements are true; they normalize the same ``\hat p`` by different
things. The ``x``-momentum flux is the raw moment

```math
M_{200}=\int v_x^2 f\,d^3v = n\!\left(u_x^2+\tfrac{T}{2}\right),
```

so the pressure appearing *inside the moment equations* is ``nT/2``. That is
``\hat p`` measured in units of the momentum-flux scale
``m\,n_\mathrm{ref}V_\mathrm{ref}^2``, whereas ``p=nT`` is ``\hat p`` measured
in units of ``p_\mathrm{ref}=n_\mathrm{ref}k_BT_\mathrm{ref}``. The two scales
differ by exactly two,

```math
m\,n_\mathrm{ref}V_\mathrm{ref}^2 = 2\,n_\mathrm{ref}k_BT_\mathrm{ref}
                                  = 2\,p_\mathrm{ref},
```

because this page fixes ``V_\mathrm{ref}=\sqrt{2k_BT_\mathrm{ref}/m}`` (to make
the Maxwellian exponent ``\exp(-|\mathbf v-\mathbf u|^2/T)``) while defining
``p_\mathrm{ref}`` without the 2. It is the same factor as the ``T/2``
per-direction variance noted above, and nothing in the code mixes the two.

The compact way to remember it: **in code units the gas constant is
``R=1/2``**, i.e. ``p_\text{mech}=nRT``. Consequences, for a monatomic gas
(``\gamma=5/3``):

| quantity | code units | trap |
|---|---|---|
| sound speed | ``c=\sqrt{\gamma R T}=\sqrt{5T/6}`` | **not** ``\sqrt{\gamma T}`` (off by ``\sqrt2``) |
| dynamic viscosity in the momentum equation | ``\mu_c=\tau\,p_\text{mech}=\tfrac12\texttt{mu\_sref}\,T^{\omega}`` | **not** ``\texttt{mu\_sref}\,T^{\omega}`` |
| Reynolds number | ``\mathrm{Re}=nuL/\mu_c=2nu/(\texttt{mu\_sref}\,T^{\omega})`` | **not** ``nu/\texttt{mu\_sref}`` |

The viscosity row is the same factor once more: the boxed conversion in the BGK
section below, ``\hat\mu=\mu_\mathrm{ref}\,\texttt{mu\_sref}\,T^{\omega}`` with
``\mu_\mathrm{ref}=p_\mathrm{ref}t_\mathrm{ref}``, is the correct way to get
Pa·s, but the natural momentum-flux viscosity scale is
``\rho_\mathrm{ref}V_\mathrm{ref}L_\mathrm{ref}=2\mu_\mathrm{ref}``. Writing the
1D Navier–Stokes momentum flux as
``n u^2+nT/2+\sigma_{xx}`` with ``\sigma_{xx}=-\tfrac43\mu_c\,\partial_x u_x``
requires ``\mu_c``.

``\tau`` and ``\mathrm{Kn}`` are untouched by all of this — a time and a length
ratio have no pressure scale in them — which is why [`tau_BGK`](@ref) and
[`Knudsen_number`](@ref) below need no factor of two.

## BGK collision time, viscosity and Knudsen number

[`tau_BGK`](@ref) returns the code relaxation time

```math
\tau=\frac{\mu_\text{code}}{p}=\texttt{mu\_sref}\;\frac{T^{\,\omega-1}}{n},
```

with the VHS viscosity law ``\mu_\text{code}=\texttt{mu\_sref}\,T^{\omega}`` and
``p=nT``. Here ``\omega`` (`omega`) is the viscosity–temperature exponent
(``\omega=1/2`` hard spheres, ``\omega=1`` Maxwell molecules) and `mu_sref` is the
**code** viscosity at ``n=T=1``. Note that ``\tau=\texttt{mu\_sref}`` requires
``n=1`` as well as ``T=1``, not ``T=1`` alone.

!!! warning "Which `T` to pass"
    The `T` in the formula above is the code temperature defined in the
    [Moments](#Moments) section, ``T=\frac{2}{3}(\langle v^2\rangle-|\mathbf u|^2)``
    — the same `T` that parameterizes `exp(-|v-u|²/T)`. It is *not*
    ``\frac{1}{3}(\langle v^2\rangle-|\mathbf u|^2)``.

    Both solvers pass the **matched Maxwellian's own** `mb_T` from
    [`find_MB_solution_T_and_v!`](@ref). That is this same quantity, and it is
    additionally consistent with the truncated velocity grid, where the raw-moment
    formula carries a small quadrature/tail error (a few times ``10^{-4}`` for
    cold, fast-drifting states). No rescaling happens inside [`tau_BGK`](@ref) or
    [`Knudsen_number`](@ref) — `mu_sref` and `T` are used as-is.

Converting the collision scale to/from physical units:

* physical relaxation time ``\hat\tau=t_\mathrm{ref}\,\tau``;
* physical viscosity ``\hat\mu=\mu_\mathrm{ref}\,\mu_\text{code}
  =\dfrac{n_\mathrm{ref}k_BT_\mathrm{ref}L_\mathrm{ref}}{V_\mathrm{ref}}\,
  \texttt{mu\_sref}\,T^{\omega}``.

So to reproduce a gas of known viscosity ``\hat\mu(T_\mathrm{ref})`` set

```math
\boxed{\;\texttt{mu\_sref}=\frac{\hat\mu(T_\mathrm{ref})}{\mu_\mathrm{ref}}
      =\frac{\hat\mu(T_\mathrm{ref})\,V_\mathrm{ref}}
             {n_\mathrm{ref}k_BT_\mathrm{ref}L_\mathrm{ref}}\;}
```

Because the mean free path scales as ``\lambda\sim\mu/p\times v_\text{th}``, in code
units ``\lambda\sim\texttt{mu\_sref}`` at the reference state and ``L_\text{code}=1``,
so **`mu_sref` is the Knudsen number up to an ``\mathcal O(1)`` constant** set by the
precise mean-free-path definition (see below). Larger `mu_sref` ⇒ more rarefied.

## Velocity grid and CFL

The grid spans ``v_i\in[-\texttt{extent},\texttt{extent}]`` per axis, i.e. physical
``\hat c_i\in[-\texttt{extent}\cdot V_\mathrm{ref},\,\texttt{extent}\cdot V_\mathrm{ref}]``.
`extent` must comfortably cover ``\max|\mathbf u|+\text{several}\times\sqrt{T}``
(a few thermal speeds beyond the fastest bulk velocity), or the cut-off Maxwellian
loses mass/energy.

The transport time step uses `max_eigenvalue = extent` (the largest ``|v_x|`` on the
grid): ``dt=\texttt{CFL}\cdot dx/\texttt{extent}`` in code units;
the physical step is ``\hat{dt}=t_\mathrm{ref}\,dt``.

## Recipe

**Physical → code inputs** (for [`run_1d`](@ref)): pick ``m``, ``n_\mathrm{ref}``,
``T_\mathrm{ref}``, ``L_\mathrm{ref}``; form ``V_\mathrm{ref},t_\mathrm{ref}`` etc.
Then

| physical state | code input |
|----------------|------------|
| number density ``\hat n`` | `n = n̂ / n_ref` |
| bulk velocity ``\hat U_x`` | `ux = Û_x / V_ref` |
| temperature ``\hat T`` | `T = T̂ / T_ref` |
| domain ``[\hat x_\text{min},\hat x_\text{max}]`` | `x_min,x_max = x̂ / L_ref` |
| end time ``\hat t_\text{end}`` | `t_end = t̂_end / t_ref` |
| viscosity ``\hat\mu`` | `mu_sref` via the boxed formula |

**Code → physical outputs**: ``\hat n=n\,n_\mathrm{ref}``,
``\hat U=u\,V_\mathrm{ref}``, ``\hat T=T\,T_\mathrm{ref}``,
``\hat p=nT\,p_\mathrm{ref}``, ``\hat t=t\,t_\mathrm{ref}``, and general moments via
``\mathcal M_{abc}=n_\mathrm{ref}V_\mathrm{ref}^{\,a+b+c}M_{abc}``.

### Worked example (N₂ at STP, 1 mm gap)

``m=4.65\times10^{-26}\,\mathrm{kg}``, ``T_\mathrm{ref}=273\,\mathrm{K}``,
``n_\mathrm{ref}=2.69\times10^{25}\,\mathrm{m^{-3}}`` (Loschmidt),
``L_\mathrm{ref}=10^{-3}\,\mathrm{m}``:

```math
V_\mathrm{ref}\approx402\ \mathrm{m/s},\quad
t_\mathrm{ref}\approx2.5\times10^{-6}\ \mathrm{s},\quad
p_\mathrm{ref}\approx1.01\times10^{5}\ \mathrm{Pa}\ (\approx1\ \text{atm}),\quad
\mu_\mathrm{ref}\approx0.25\ \mathrm{Pa\,s}.
```

Air viscosity ``\hat\mu\approx1.66\times10^{-5}\,\mathrm{Pa\,s}`` gives
``\texttt{mu\_sref}\approx6.6\times10^{-5}``, which matches the STP Knudsen number
``\lambda/L\approx68\,\mathrm{nm}/1\,\mathrm{mm}\approx6.8\times10^{-5}`` — 1 mm at
1 atm is essentially continuum, as expected.

## Knudsen number

The local (dimensionless) mean free path is, up to an ``\mathcal O(1)`` constant,
the distance a particle travels in one collision time, using the code-unit
thermal speed ``\sqrt T``:

```math
\lambda(n,T)\approx\sqrt T\,\tau(n,T)
  =\texttt{mu\_sref}\;\frac{T^{\,\omega-1/2}}{n}.
```

Since the domain length is already normalized to ``L_\text{code}=1``, this *is*
the local Knudsen number:

```math
\boxed{\;\mathrm{Kn}(n,T)=\frac{\lambda(n,T)}{L}
  =\texttt{mu\_sref}\;\frac{T^{\,\omega-1/2}}{n}\;}
```

At the reference state ``n=T=1`` this reduces to ``\mathrm{Kn}_\mathrm{ref}
=\texttt{mu\_sref}`` exactly, matching the statement above. Limits:

* ``\omega=1/2`` (hard spheres): ``\mathrm{Kn}=\texttt{mu\_sref}/n``, independent
  of ``T``;
* ``\omega=1`` (Maxwell molecules): ``\mathrm{Kn}=\texttt{mu\_sref}\sqrt T/n``.

In general ``\mathrm{Kn}`` grows with ``T`` (hotter ⇒ longer mean free path)
and shrinks with ``n`` (denser ⇒ shorter mean free path), tracking ``\tau``.

Substituting the boxed `mu_sref` relation above gives the reference-state
Knudsen number directly in physical quantities:

```math
\mathrm{Kn}_\mathrm{ref}=\frac{\hat\mu(T_\mathrm{ref})}
  {n_\mathrm{ref}L_\mathrm{ref}}\sqrt{\frac{2}{mk_BT_\mathrm{ref}}},
```

which is the standard ``\lambda/L`` scaling (confirmed numerically in the
worked example above: ``\texttt{mu\_sref}\approx6.6\times10^{-5}`` vs.
``\lambda/L\approx6.8\times10^{-5}``), and the general local form

```math
\mathrm{Kn}(n,T)=\frac{\hat\mu(T_\mathrm{ref})}{n\,n_\mathrm{ref}L_\mathrm{ref}}
  \sqrt{\frac{2}{mk_BT_\mathrm{ref}}}\;T^{\,\omega-1/2}.
```

The function [`Knudsen_number`](@ref) computes the actual Knudsen number using
the full definitions of the mean free path and viscosity, i.e.
a factor `C_omega = ((5.0 - 2.0*omega) * (7.0 - 2.0*omega)) / (5.0 * (1.0 + omega) * sqrt(2.0 * pi))`
is present. See Eqn. (4.65) in "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"
or (D.21) in "Nonequilibrium Gas Dynamics and Molecular Simulation"
