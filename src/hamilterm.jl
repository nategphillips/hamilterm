module hamilterm

using LinearAlgebra

include("constants.jl")
include("elements.jl")
include("options.jl")
include("terms.jl")
include("utils.jl")

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

function construct_n_operator_matrices(
    s_qn::Float64,
    j_qn::Float64,
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    max_n_index::Int,
    dim::Int,
)
    n_op_mats = [zeros(Float64, dim, dim) for _ in 1:6]

    n_op_mats[begin] = n_squared(s_qn, j_qn, sigma_vec, omega_vec, dim)

    for i in 2:max_n_index
        n_op_mats[i] = n_op_mats[i-1] * n_op_mats[begin]
    end

    return n_op_mats
end

function compute_hamiltonian!(comp::Computation)
    s_qn, lambda_qn = parse_term_symbol(comp.term_symbol)
    basis_fns = generate_basis_fns(s_qn, lambda_qn)
    dim = length(basis_fns)
    lambda_vec, sigma_vec, omega_vec = generate_basis_vectors(basis_fns)
    n_op_mats = construct_n_operator_matrices(s_qn, comp.j_qn, sigma_vec, omega_vec, comp.max_n_index, dim)

    h_mat = zeros(Float64, dim, dim)

    if INCLUDE_RO
        rotational!(h_mat, n_op_mats, comp.consts.rotational)
    end
    if INCLUDE_SO
        spin_orbit!(h_mat, s_qn, lambda_vec, sigma_vec, n_op_mats, comp.max_acomm_index, comp.consts.spin_orbit)
    end
    if INCLUDE_SS
        spin_spin!(h_mat, s_qn, sigma_vec, n_op_mats, comp.max_acomm_index, comp.consts.spin_spin)
    end
    if INCLUDE_SR
        spin_rotation!(h_mat, s_qn, comp.j_qn, sigma_vec, omega_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.spin_rotation)
    end
    if INCLUDE_LD
        lambda_doubling!(h_mat, s_qn, comp.j_qn, lambda_vec, sigma_vec, omega_vec, n_op_mats, dim, comp.max_acomm_index, comp.consts.lambda_doubling)
    end

    comp.hamiltonian = Hermitian(h_mat)

    return nothing
end

function compute_eigensystem!(comp::Computation)
    comp.eigenvalues, comp.eigenvectors = eigen(comp.hamiltonian)

    return nothing
end

function initialize_computation(
    term_symbol::String,
    consts::AllConsts,
    j_qn::Float64,
    max_n_index::Int,
    max_acomm_index::Int,
)
    comp = Computation(term_symbol, consts, j_qn, max_n_index, max_acomm_index)
    compute_hamiltonian!(comp)
    compute_eigensystem!(comp)

    return comp
end

function three_sigma(num::Int)
    c = AllConsts(
        rotational=RotationalConsts(B=0.8132, D=4.50e-06),
        spin_spin=SpinSpinConsts(lambda=1.69),
        spin_rotation=SpinRotationConsts(gamma=-0.028)
    )
    for _ in 0:num
        comp = initialize_computation("3Sigma", c, 1.0, 6, 4)
    end
end

function two_pi(num::Int)
    c = AllConsts(
        rotational=RotationalConsts(B=18.55),
        spin_orbit=SpinOrbitConsts(A=-139.21),
        lambda_doubling=LambdaDoublingConsts(p=0.235, q=-0.0391)
    )
    for _ in 0:num
        comp = initialize_computation("2Pi", c, 1.0, 6, 4)
    end

    return nothing
end

function five_pi(num::Int)
    c = AllConsts(
        rotational=RotationalConsts(
            B=18.55,
            D=4.50e-06,
            H=4.50e-06,
            L=4.50e-06,
            M=4.50e-06,
            P=4.50e-06,
        ),
        spin_orbit=SpinOrbitConsts(A=-139.21, A_D=1.0, A_H=1.0, A_L=1.0, A_M=1.0, eta=1.0),
        spin_spin=SpinSpinConsts(lambda=1.69, lambda_D=1.0, lambda_H=1.0, theta=1.0),
        spin_rotation=SpinRotationConsts(
            gamma=-0.028,
            gamma_D=-0.028,
            gamma_H=-0.028,
            gamma_L=-0.028,
            gamma_S=-0.028,
        ),
        lambda_doubling=LambdaDoublingConsts(
            o=0.1,
            p=0.235,
            q=-0.0391,
            o_D=0.1,
            p_D=0.1,
            q_D=0.1,
            o_H=0.1,
            p_H=0.1,
            q_H=0.1,
            o_L=0.1,
            p_L=0.1,
            q_L=0.1,
        ),
    )
    for _ in 0:num
        comp = initialize_computation("5Pi", c, 5.0, 6, 4)
    end

    return nothing
end

@time three_sigma(1000)
@time two_pi(500)
@time five_pi(200)

end # module hamilterm
