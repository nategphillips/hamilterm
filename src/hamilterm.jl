module hamilterm

using LinearAlgebra

include("constants.jl")
include("elements.jl")
include("options.jl")
include("terms.jl")
include("utils.jl")

# TODO: 1) Pass around a single Hamiltonian matrix instead of creating new ones in each function
#       2) Spin-orbit has 4 centrifugal distortion constants, while everything else has 3

mutable struct Computation
    term_symbol::String
    consts::AllConsts
    j_qn::Float64
    max_n_index::Int
    max_acomm_index::Int
    hamiltonian::Hermitian{Float64,Matrix{Float64}}
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}

    Computation(term_symbol, consts, j_qn, max_n_index, max_acomm_index) = new(term_symbol, consts, j_qn, max_n_index, max_acomm_index)
end

function construct_n_operator_matrices(s_qn, j_qn, sigma_vec, omega_vec, max_n_index, dim)
    n_op_mats = [zeros(Float64, dim, dim) for _ in 1:6]

    n_op_mats[begin] = n_squared(s_qn, j_qn, sigma_vec, omega_vec, dim)

    for i in 2:max_n_index
        n_op_mats[i] = n_op_mats[i-1] * n_op_mats[begin]
    end

    n_op_mats
end

function compute_hamiltonian!(comp::Computation)
    s_qn, lambda_qn = parse_term_symbol(comp.term_symbol)
    basis_fns = generate_basis_fns(s_qn, lambda_qn)
    dim = length(basis_fns)
    lambda_vec, sigma_vec, omega_vec = generate_basis_vectors(basis_fns)
    n_op_mats = construct_n_operator_matrices(s_qn, comp.j_qn, sigma_vec, omega_vec, comp.max_n_index, dim)

    h_mat = zeros(Float64, dim, dim)

    if INCLUDE_RO
        h_mat .+= rotational(n_op_mats, comp.consts.rotational)
    end
    if INCLUDE_SO
        h_mat .+= spin_orbit(s_qn, lambda_vec, sigma_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.spin_orbit)
    end
    if INCLUDE_SS
        h_mat .+= spin_spin(s_qn, sigma_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.spin_spin)
    end
    if INCLUDE_SR
        h_mat .+= spin_rotation(s_qn, comp.j_qn, sigma_vec, omega_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.spin_rotation)
    end
    if INCLUDE_LD
        h_mat .+= lambda_doubling(s_qn, comp.j_qn, lambda_vec, sigma_vec, omega_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.lambda_doubling)
    end

    comp.hamiltonian = Hermitian(h_mat)
end

function compute_eigensystem!(comp::Computation)
    comp.eigenvalues, comp.eigenvectors = eigen(comp.hamiltonian)
end

function initialize_computation(term_symbol::String, consts::AllConsts, j_qn::Float64, max_n_index::Int, max_acomm_index::Int)
    comp = Computation(term_symbol, consts, j_qn, max_n_index, max_acomm_index)
    compute_hamiltonian!(comp)
    compute_eigensystem!(comp)
    comp
end

function two_pi(num::Int)
    c = AllConsts(
        rotational=RotationalConsts(B=18.55),
        spin_orbit=SpinOrbitConsts(A=-139.21),
        lambda_doubling=LambdaDoublingConsts(p=0.235, q=-0.0391)
    )
    for _ in 0:num
        comp = initialize_computation("2Pi", c, 10.0, 6, 3)
    end
end

@time two_pi(500)

end # module hamilterm
