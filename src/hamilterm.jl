module hamilterm

export initialize_computation,
    AllConsts,
    RotationalConsts,
    SpinOrbitConsts,
    SpinSpinConsts,
    SpinRotationConsts,
    LambdaDoublingConsts

using LinearAlgebra

include("constants.jl")
include("elements.jl")
include("options.jl")
include("terms.jl")
include("utils.jl")

mutable struct Computation
    term_symbol::String
    consts::AllConsts
    J::Float64
    max_n_index::Int
    max_acomm_index::Int
    hamiltonian::Hermitian{Float64,Matrix{Float64}}
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}

    function Computation(term_symbol, consts, J, max_n_index, max_acomm_index)
        return new(term_symbol, consts, J, max_n_index, max_acomm_index)
    end
end

function construct_n_operator_matrices(
    S::Float64,
    J::Float64,
    Σ_vec::Vector{Float64},
    Ω_vec::Vector{Float64},
    max_n_index::Int,
    dim::Int,
)
    n_op_mats = [zeros(Float64, dim, dim) for _ in 1:6]

    n_op_mats[begin] = n_squared(S, J, Σ_vec, Ω_vec, dim)

    for i in 2:max_n_index
        n_op_mats[i] = n_op_mats[i - 1] * n_op_mats[begin]
    end

    return n_op_mats
end

function compute_hamiltonian!(comp::Computation)
    S, Λ = parse_term_symbol(comp.term_symbol)
    basis_fns = generate_basis_fns(S, Λ)
    dim = length(basis_fns)
    Λ_vec, Σ_vec, Ω_vec = generate_basis_vectors(basis_fns)
    n_op_mats = construct_n_operator_matrices(S, comp.J, Σ_vec, Ω_vec, comp.max_n_index, dim)

    H_mat = zeros(Float64, dim, dim)

    if INCLUDE_RO
        rotational!(H_mat, n_op_mats, comp.consts.rotational)
    end
    if INCLUDE_SO
        spin_orbit!(H_mat, S, Λ_vec, Σ_vec, n_op_mats, comp.max_acomm_index, comp.consts.spin_orbit)
    end
    if INCLUDE_SS
        spin_spin!(H_mat, S, Σ_vec, n_op_mats, comp.max_acomm_index, comp.consts.spin_spin)
    end
    if INCLUDE_SR
        spin_rotation!(
            H_mat,
            S,
            comp.J,
            Σ_vec,
            Ω_vec,
            n_op_mats,
            dim,
            comp.max_acomm_index,
            comp.consts.spin_rotation,
        )
    end
    if INCLUDE_LD
        lambda_doubling!(
            H_mat,
            S,
            comp.J,
            Λ_vec,
            Σ_vec,
            Ω_vec,
            n_op_mats,
            dim,
            comp.max_acomm_index,
            comp.consts.lambda_doubling,
        )
    end

    comp.hamiltonian = Hermitian(H_mat)

    return nothing
end

function compute_eigensystem!(comp::Computation)
    comp.eigenvalues, comp.eigenvectors = eigen(comp.hamiltonian)

    return nothing
end

function initialize_computation(
    term_symbol::String, consts::AllConsts, J::Float64, max_n_index::Int, max_acomm_index::Int
)
    comp = Computation(term_symbol, consts, J, max_n_index, max_acomm_index)
    compute_hamiltonian!(comp)
    compute_eigensystem!(comp)

    return comp
end

end # module hamilterm
