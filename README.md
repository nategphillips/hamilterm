# hamilterm

Symbolically computes the rotational Hamiltonian for a given molecular term symbol.

## Example Output

Sample output for a $^2\Sigma$ state is shown below. See `./docs/main.pdf` to avoid GitHub's poor excuse for $\LaTeX$ formatting.

**Info:**

- Computed up to $\mathbf{N}^4$
- Max anticommutator value $\mathbf{N}^2$

**Term symbol:**

- $^2\Sigma: \quad S=1/2,\;\Lambda=0$

**Basis states $\lvert \Lambda,\Sigma,\Omega\rangle$:**

- $\lvert \pm0,\,-1/2,\,-1/2\rangle$
- $\lvert \pm0,\,+1/2,\,+1/2\rangle$

**Hamiltonian matrix:**

$$
\begin{bmatrix}B x + \frac{B}{4} - D x^{2} - \frac{3 D x}{2} - \frac{5 D}{16} - \frac{\gamma}{2} - \gamma_{D} x - \frac{\gamma_{D}}{4} & \frac{\sqrt{4 x + 1} \left(- 8 B + 4 D \left(4 x + 1\right) + 4 \gamma + \gamma_{D} \left(4 x + 5\right)\right)}{16} \\
\frac{\sqrt{4 x + 1} \left(- 8 B + 4 D \left(4 x + 1\right) + 4 \gamma + \gamma_{D} \left(4 x + 5\right)\right)}{16} & B x + \frac{B}{4} - D x^{2} - \frac{3 D x}{2} - \frac{5 D}{16} - \frac{\gamma}{2} - \gamma_{D} x - \frac{\gamma_{D}}{4}\end{bmatrix}
$$

**Eigenvalues:**

$$
\begin{aligned}
F_{1} &= B x + \frac{B}{4} - D x^{2} - \frac{3 D x}{2} - \frac{5 D}{16} - \frac{\gamma}{2} - \gamma_{D} x - \frac{\gamma_{D}}{4} - \sqrt{4 x + 1} \left(- \frac{B}{2} + D x + \frac{D}{4} + \frac{\gamma}{4} + \frac{\gamma_{D} x}{4} + \frac{5 \gamma_{D}}{16}\right) \\
F_{2} &= B x + \frac{B}{4} - D x^{2} - \frac{3 D x}{2} - \frac{5 D}{16} - \frac{\gamma}{2} - \gamma_{D} x - \frac{\gamma_{D}}{4} + \sqrt{4 x + 1} \left(- \frac{B}{2} + D x + \frac{D}{4} + \frac{\gamma}{4} + \frac{\gamma_{D} x}{4} + \frac{5 \gamma_{D}}{16}\right) \\
\end{aligned}
$$

## Roadmap

Computing and simplifying the eigenvalues for $^2\Pi$ and higher states is extremely slow with SymPy. Some options for future speedups include:

- Replacing SymPy with SymEngine.
- Switching to Julia and using Symbolics.jl.

## License and Copyright

Copyright (C) 2025 Nathan G. Phillips

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
