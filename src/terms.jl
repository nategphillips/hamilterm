function rotational!(
    ham::Matrix{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    r_consts::RotationalConsts,
)
    ham .+= (
        r_consts.B .* n_op_mats[1]
        -
        r_consts.D .* n_op_mats[2]
        + r_consts.H .* n_op_mats[3]
        + r_consts.L .* n_op_mats[4]
        + r_consts.M .* n_op_mats[5]
        + r_consts.P .* n_op_mats[6]
    )

    return nothing
end

function spin_orbit!(
    ham::Matrix{Float64},
    s_qn::Float64,
    lambda_vec::Vector{Int},
    sigma_vec::Vector{Float64},
    n_op_mats,
    max_acomm_index::Int,
    so_consts::SpinOrbitConsts,
)
    if maximum(abs.(lambda_vec)) == 0.0 || s_qn <= 0.0
        return nothing
    end

    lzsz = lz_sz(lambda_vec, sigma_vec)
    ham .+= so_consts.A * lzsz

    vec = [so_consts.A_D, so_consts.A_H, so_consts.A_L, so_consts.A_M]
    spin_orbit_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_orbit_cd_consts)
        for (idx, constant) in enumerate(spin_orbit_cd_consts)
            ham .+= 0.5 * constant * (n_op_mats[idx] * lzsz + lzsz * n_op_mats[idx])
        end
    end

    if s_qn > 1.0
        ham .+= so_consts.eta * lzsz * (sigma_vec^2 - 0.2 * (3 * s_squared(s_qn) - 1))
    end

    return nothing
end

function spin_spin!(
    ham::Matrix{Float64},
    s_qn::Float64,
    sigma_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    max_acomm_index::Int,
    ss_consts::SpinSpinConsts,
)
    if s_qn <= 0.5
        return nothing
    end

    tsms = three_sz2_minus_s2(s_qn, sigma_vec)
    ham .+= (2.0 * ss_consts.lambda / 3.0) * tsms

    vec = [ss_consts.lambda_D, ss_consts.lambda_H]
    spin_spin_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_spin_cd_consts)
        for (idx, constant) in enumerate(spin_spin_cd_consts)
            ham .+= (constant / 3.0) * (tsms * n_op_mats[idx] + n_op_mats[idx] * tsms)
        end
    end

    if s_qn > 1.5
        ham .+= diagm(
            (ss_consts.theta / 12.0) * (
                35.0 * sigma_vec^4
                -
                30.0 * s_squared(s_qn)
                +
                25.0 * sigma_vec^2
                -
                6.0 * s_squared(s_qn)
                +
                3.0 * s_squared(s_qn)^2
            )
        )
    end

    return nothing
end

function spin_rotation!(
    ham::Matrix{Float64},
    s_qn::Float64,
    j_qn::Float64,
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    dim::Int,
    max_acomm_index::Int,
    sr_consts::SpinRotationConsts,
)
    if s_qn <= 0.0
        return nothing
    end

    ndots = n_dot_s(s_qn, j_qn, sigma_vec, omega_vec, dim)
    ham .+= sr_consts.gamma * ndots

    vec = [sr_consts.gamma_D, sr_consts.gamma_H, sr_consts.gamma_L]
    spin_rotation_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_rotation_cd_consts)
        for (idx, constant) in enumerate(spin_rotation_cd_consts)
            ham .+= 0.5 * constant * (ndots * n_op_mats[idx] + n_op_mats[idx] * ndots)
        end
    end

    if s_qn > 1.0
        sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
        omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

        mask_plus = @. (sigma_i == sigma_j - 1) & (omega_i == omega_j - 1)
        mask_minus = @. (sigma_i == sigma_j + 1) & (omega_i == omega_j + 1)

        common_term = -0.5 * sr_consts.gamma_S * (s_squared(s_qn) - 5.0 * sigma_j * (sigma_j - 1.0) - 2.0)

        term_plus = common_term * j_plus(j_qn, omega_j) * s_minus(s_qn, sigma_j)
        term_minus = common_term * j_minus(j_qn, omega_j) * s_plus(s_qn, sigma_j)

        ham[mask_plus] .+= term_plus[mask_plus]
        ham[mask_minus] .+= term_minus[mask_minus]
    end

    return nothing
end

function lambda_doubling!(
    ham::Matrix{Float64},
    s_qn::Float64,
    j_qn::Float64,
    lambda_vec::Vector{Int},
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    dim::Int,
    max_acomm_index::Int,
    ld_consts::LambdaDoublingConsts,
)
    if maximum(lambda_vec) != 1.0
        return nothing
    end

    sp2sm2 = sp2_plus_sm2(s_qn, lambda_vec, sigma_vec, dim)
    ham .+= 0.5 * (ld_consts.o + ld_consts.p + ld_consts.q) * sp2sm2

    jpspjmsm = jpsp_plus_jmsm(s_qn, j_qn, lambda_vec, sigma_vec, omega_vec, dim)
    ham .-= 0.5 * (ld_consts.p + 2.0 * ld_consts.q) * jpspjmsm

    jp2jm2 = jp2_plus_jm2(j_qn, lambda_vec, omega_vec, dim)
    ham .+= 0.5 * ld_consts.q * jp2jm2

    vec1 = [
        ld_consts.o_D + ld_consts.p_D + ld_consts.q_D,
        ld_consts.o_H + ld_consts.p_H + ld_consts.q_H,
        ld_consts.o_L + ld_consts.p_L + ld_consts.q_L,
    ]
    lambda_doubling_cd_consts_opq = safe_slice(vec1, max_acomm_index)

    vec2 = [
        ld_consts.p_D + 2.0 * ld_consts.q_D,
        ld_consts.p_H + 2.0 * ld_consts.q_H,
        ld_consts.p_L + 2.0 * ld_consts.q_L,
    ]
    lambda_doubling_cd_consts_pq = safe_slice(vec2, max_acomm_index)

    vec3 = [
        ld_consts.q_D,
        ld_consts.q_H,
        ld_consts.q_L,
    ]
    lambda_doubling_cd_consts_q = safe_slice(vec3, max_acomm_index)

    if any(!iszero, lambda_doubling_cd_consts_opq + lambda_doubling_cd_consts_pq + lambda_doubling_cd_consts_q)
        for (idx, constant) in enumerate(lambda_doubling_cd_consts_opq)
            ham .+= 0.25 * constant * (sp2sm2 * n_op_mats[idx] + n_op_mats[idx] * sp2sm2)
        end

        for (idx, constant) in enumerate(lambda_doubling_cd_consts_pq)
            ham .-= 0.25 * constant * (jpspjmsm * n_op_mats[idx] + n_op_mats[idx] * jpspjmsm)
        end

        for (idx, constant) in enumerate(lambda_doubling_cd_consts_q)
            ham .+= 0.25 * constant * (jp2jm2 * n_op_mats[idx] + n_op_mats[idx] * jp2jm2)
        end
    end

    return nothing
end
