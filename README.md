# Can solitons model tsunamis?

Finite-difference simulation of the Korteweg–de Vries (KdV) equation in Python,
written for my TIPE (French preparatory-class research project).

- Two time-stepping schemes: Runge–Kutta 4 and leapfrog, with periodic boundaries
- Reproduces soliton propagation, two-soliton interaction, and a Gaussian
  breaking up into a train of solitons
- Validated against the exact soliton solution: second-order convergence in space

Conclusion: the soliton model does not apply to the 2004 Indian Ocean tsunami.
The wave travelled about 1,500 km, far less than the roughly 9,000 km the KdV
regime needs to develop.

Run `simulations.py` for the demos and `tests.py` for the convergence tests
(requires numpy and matplotlib).
