module hamilterm

export compute_eigensystem,
    Params,
    Consts,
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

# TODO: 25/08/13
#   ✓ Configure formatting to follow BlueStyle
#   × Add package tests and benchmarks
#   × Ensure type stability with JET.jl / Cthulhu.jl / DispatchDoctor.jl
#   × Use StaticArrays.jl since all arrays & matrices are ≤ 100 elements
#   × Add automated checks using Aqua.jl
#   × Write docstrings

struct Params
    term_symbol::String
    J::Float64
    max_n_index::Int
    max_acomm_index::Int
end

function compute_hamiltonian(params::Params, consts::Consts)
    S, Λ = parse_term_symbol(params.term_symbol)
    basis_fns = generate_basis_fns(S, Λ)
    dim = length(basis_fns)
    Λ_vec, Σ_vec, Ω_vec = generate_basis_vectors(basis_fns)
    n_op_mats = construct_n_operator_matrices(S, params.J, Σ_vec, Ω_vec, params.max_n_index, dim)

    H_mat = zeros(Float64, dim, dim)

    if INCLUDE_RO
        rotational!(H_mat, n_op_mats, consts.rotational)
    end
    if INCLUDE_SO
        spin_orbit!(H_mat, S, Λ_vec, Σ_vec, n_op_mats, params.max_acomm_index, consts.spin_orbit)
    end
    if INCLUDE_SS
        spin_spin!(H_mat, S, Σ_vec, n_op_mats, params.max_acomm_index, consts.spin_spin)
    end
    if INCLUDE_SR
        spin_rotation!(
            H_mat,
            S,
            params.J,
            Σ_vec,
            Ω_vec,
            n_op_mats,
            dim,
            params.max_acomm_index,
            consts.spin_rotation,
        )
    end
    if INCLUDE_LD
        lambda_doubling!(
            H_mat,
            S,
            params.J,
            Λ_vec,
            Σ_vec,
            Ω_vec,
            n_op_mats,
            dim,
            params.max_acomm_index,
            consts.lambda_doubling,
        )
    end

    return Hermitian(H_mat)
end

function compute_eigensystem(params::Params, consts::Consts)
    return eigen(compute_hamiltonian(params, consts))
end

end # module hamilterm
