function parse_term_symbol(term_symbol::String)
    spin_multiplicity = parse(Int, term_symbol[begin])
    S = 0.5 * (spin_multiplicity - 1)
    state = term_symbol[(begin + 1):end]
    Λ = LAMBDA_INT_MAP[state]

    return S, Λ
end

function generate_basis_fns(S::Float64, Λ::Int)
    Σ_vals = (-S):1:S
    # The first Λ is wrapped in a 1-tuple since both returns need to be iterable.
    Λ_vals = Λ == 0 ? (Λ,) : (-Λ, Λ)

    return [(Λ, Σ, Λ + Σ) for Λ in Λ_vals for Σ in Σ_vals]
end

function generate_basis_vectors(basis_fns::Vector{Tuple{Int,Float64,Float64}})
    Λ_vec = [Λ for (Λ, _, _) in basis_fns]
    Σ_vec = [Σ for (_, Σ, _) in basis_fns]
    Ω_vec = [Ω for (_, _, Ω) in basis_fns]

    return Λ_vec, Σ_vec, Ω_vec
end

function generate_basis_matrices(basis_vec::Vector{<:Real}, dim::Int)
    basis_mat_i = repeat(basis_vec, 1, dim)
    basis_mat_j = basis_mat_i'

    return basis_mat_i, basis_mat_j
end

function safe_slice(vec::Vector{Float64}, n::Int)
    return vec[begin:min(length(vec), n)]
end
