function parse_term_symbol(term_symbol::String)
    spin_multiplicity = parse(Int, term_symbol[begin])
    s_qn = 0.5 * (spin_multiplicity - 1)
    state = term_symbol[begin+1:end]
    lambda_qn = LAMBDA_INT_MAP[state]
    s_qn, lambda_qn
end

function generate_basis_fns(s_qn::Float64, lambda_qn::Int)
    Σ_vals = -s_qn+0:2*s_qn
    Λ_vals = lambda_qn == 0 ? [lambda_qn] : [-lambda_qn, lambda_qn]

    [(Λ, Σ, Λ + Σ) for Λ in Λ_vals for Σ in Σ_vals]
end

function generate_basis_vectors(basis_fns)
    Λ_basis = [Λ for (Λ, _, _) in basis_fns]
    Σ_basis = [Σ for (_, Σ, _) in basis_fns]
    Ω_basis = [Ω for (_, _, Ω) in basis_fns]
    Λ_basis, Σ_basis, Ω_basis
end

function generate_basis_matrices(basis_vec, dim)
    basis_matrix_i = repeat(basis_vec, 1, dim)
    basis_matrix_j = basis_matrix_i'
    basis_matrix_i, basis_matrix_j
end
