function rotational!(
    H_mat::Matrix{Float64}, n_op_mats::Vector{Matrix{Float64}}, r_consts::RotationalConsts
)
    H_mat .+= (
        r_consts.B .* n_op_mats[1] - r_consts.D .* n_op_mats[2] +
        r_consts.H .* n_op_mats[3] +
        r_consts.L .* n_op_mats[4] +
        r_consts.M .* n_op_mats[5] +
        r_consts.P .* n_op_mats[6]
    )

    return nothing
end

function spin_orbit!(
    H_mat::Matrix{Float64},
    S::Float64,
    Λ_vec::Vector{Int},
    Σ_vec::Vector{Float64},
    n_op_mats,
    max_acomm_index::Int,
    so_consts::SpinOrbitConsts,
)
    if maximum(abs.(Λ_vec)) == 0.0 || S <= 0.0
        return nothing
    end

    LzSz = lz_sz(Λ_vec, Σ_vec)
    H_mat .+= so_consts.A * LzSz

    vec = [so_consts.A_D, so_consts.A_H, so_consts.A_L, so_consts.A_M]
    spin_orbit_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_orbit_cd_consts)
        for (idx, constant) in enumerate(spin_orbit_cd_consts)
            H_mat .+= 0.5 * constant * (n_op_mats[idx] * LzSz + LzSz * n_op_mats[idx])
        end
    end

    if S > 1.0
        # Since LzSz is a diagonal matrix, Σ_vec is multiplied element-wise along the diagonal.
        H_mat .+= so_consts.η * LzSz .* (Σ_vec .^ 2 .- 0.2 * (3.0 * s_squared(S) - 1.0))
    end

    return nothing
end

function spin_spin!(
    H_mat::Matrix{Float64},
    S::Float64,
    Σ_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    max_acomm_index::Int,
    ss_consts::SpinSpinConsts,
)
    if S <= 0.5
        return nothing
    end

    three_Sz²_minus_S² = three_sz2_minus_s2(S, Σ_vec)
    H_mat .+= (2.0 * ss_consts.λ / 3.0) * three_Sz²_minus_S²

    vec = [ss_consts.λ_D, ss_consts.λ_H]
    spin_spin_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_spin_cd_consts)
        for (idx, constant) in enumerate(spin_spin_cd_consts)
            H_mat .+=
                (constant / 3.0) *
                (three_Sz²_minus_S² * n_op_mats[idx] + n_op_mats[idx] * three_Sz²_minus_S²)
        end
    end

    if S > 1.5
        H_mat .+= diagm(
            (ss_consts.θ / 12.0) * (
                35.0 * Σ_vec .^ 4 .- 30.0 * s_squared(S) .+ 25.0 * Σ_vec .^ 2 .-
                6.0 * s_squared(S) .+ 3.0 * s_squared(S)^2
            ),
        )
    end

    return nothing
end

function spin_rotation!(
    H_mat::Matrix{Float64},
    S::Float64,
    J::Float64,
    Σ_vec::Vector{Float64},
    Ω_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    dim::Int,
    max_acomm_index::Int,
    sr_consts::SpinRotationConsts,
)
    if S <= 0.0
        return nothing
    end

    N_dot_S = n_dot_s(S, J, Σ_vec, Ω_vec, dim)
    H_mat .+= sr_consts.γ * N_dot_S

    vec = [sr_consts.γ_D, sr_consts.γ_H, sr_consts.γ_L]
    spin_rotation_cd_consts = safe_slice(vec, max_acomm_index)

    if any(!iszero, spin_rotation_cd_consts)
        for (idx, constant) in enumerate(spin_rotation_cd_consts)
            H_mat .+= 0.5 * constant * (N_dot_S * n_op_mats[idx] + n_op_mats[idx] * N_dot_S)
        end
    end

    if S > 1.0
        Σ_mat_i, Σ_mat_j = generate_basis_matrices(Σ_vec, dim)
        Ω_mat_i, Ω_mat_j = generate_basis_matrices(Ω_vec, dim)

        mask_plus = @. (Σ_mat_i == Σ_mat_j - 1.0) & (Ω_mat_i == Ω_mat_j - 1.0)
        mask_minus = @. (Σ_mat_i == Σ_mat_j + 1.0) & (Ω_mat_i == Ω_mat_j + 1.0)

        term_plus = (
            -0.5 * sr_consts.γ_S * (s_squared(S) .- 5.0 * Σ_mat_j .* (Σ_mat_j .- 1.0) .- 2.0) .*
            j_plus(J, Ω_mat_j) .* s_minus(S, Σ_mat_j)
        )
        term_minus = (
            -0.5 * sr_consts.γ_S * (s_squared(S) .- 5.0 * Σ_mat_j .* (Σ_mat_j .+ 1.0) .- 2.0) .*
            j_minus(J, Ω_mat_j) .* s_plus(S, Σ_mat_j)
        )

        H_mat[mask_plus] .+= term_plus[mask_plus]
        H_mat[mask_minus] .+= term_minus[mask_minus]
    end

    return nothing
end

function lambda_doubling!(
    H_mat::Matrix{Float64},
    S::Float64,
    J::Float64,
    Λ_vec::Vector{Int},
    Σ_vec::Vector{Float64},
    Ω_vec::Vector{Float64},
    n_op_mats::Vector{Matrix{Float64}},
    dim::Int,
    max_acomm_index::Int,
    ld_consts::LambdaDoublingConsts,
)
    if maximum(Λ_vec) != 1.0
        return nothing
    end

    S₊²_plus_S₋² = sp2_plus_sm2(S, Λ_vec, Σ_vec, dim)
    H_mat .+= 0.5 * (ld_consts.o + ld_consts.p + ld_consts.q) * S₊²_plus_S₋²

    J₊S₋_plus_J₋S₊ = jpsp_plus_jmsm(S, J, Λ_vec, Σ_vec, Ω_vec, dim)
    H_mat .-= 0.5 * (ld_consts.p + 2.0 * ld_consts.q) * J₊S₋_plus_J₋S₊

    J₊²_plus_J₋² = jp2_plus_jm2(J, Λ_vec, Ω_vec, dim)
    H_mat .+= 0.5 * ld_consts.q * J₊²_plus_J₋²

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

    vec3 = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]
    lambda_doubling_cd_consts_q = safe_slice(vec3, max_acomm_index)

    if any(
        !iszero,
        lambda_doubling_cd_consts_opq + lambda_doubling_cd_consts_pq + lambda_doubling_cd_consts_q,
    )
        for (idx, constant) in enumerate(lambda_doubling_cd_consts_opq)
            H_mat .+=
                0.25 * constant * (S₊²_plus_S₋² * n_op_mats[idx] + n_op_mats[idx] * S₊²_plus_S₋²)
        end

        for (idx, constant) in enumerate(lambda_doubling_cd_consts_pq)
            H_mat .-=
                0.25 *
                constant *
                (J₊S₋_plus_J₋S₊ * n_op_mats[idx] + n_op_mats[idx] * J₊S₋_plus_J₋S₊)
        end

        for (idx, constant) in enumerate(lambda_doubling_cd_consts_q)
            H_mat .+=
                0.25 * constant * (J₊²_plus_J₋² * n_op_mats[idx] + n_op_mats[idx] * J₊²_plus_J₋²)
        end
    end

    return nothing
end
