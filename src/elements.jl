s_squared(S::Float64) = S * (S + 1.0)

function s_plus(S::Float64, Σ_mat::AbstractMatrix{Float64})
    arg = S * (S + 1.0) .- Σ_mat .* (Σ_mat .+ 1.0)
    # Check for negative elements, not sure if this is the most idiomatic way of doing this.
    if any(arg .< 0.0)
        return zero(Σ_mat)
    end

    return sqrt.(arg)
end

function s_minus(S::Float64, Σ_mat::AbstractMatrix{Float64})
    arg = S * (S + 1.0) .- Σ_mat .* (Σ_mat .- 1.0)
    if any(arg .< 0.0)
        return zero(Σ_mat)
    end

    return sqrt.(arg)
end

j_squared(J::Float64) = J * (J + 1.0)

function j_plus(J::Float64, Ω_mat::AbstractMatrix{Float64})
    arg = J * (J + 1.0) .- Ω_mat .* (Ω_mat .- 1.0)
    if any(arg .< 0.0)
        return zero(Ω_mat)
    end

    return sqrt.(arg)
end

function j_minus(J::Float64, Ω_mat::AbstractMatrix{Float64})
    arg = J * (J + 1.0) .- Ω_mat .* (Ω_mat .+ 1.0)
    if any(arg .< 0.0)
        return zero(Ω_mat)
    end

    return sqrt.(arg)
end

lz_sz(Λ_vec::Vector{Int}, Σ_vec::Vector{Float64}) = diagm(Λ_vec .* Σ_vec)

function three_sz2_minus_s2(S::Float64, Σ_vec::Vector{Float64})
    return diagm(3.0 * Σ_vec .^ 2 .- s_squared(S))
end

function n_squared(S::Float64, J::Float64, Σ_vec::Vector{Float64}, Ω_vec::Vector{Float64}, dim::Int)
    Σ_mat_i, Σ_mat_j = generate_basis_matrices(Σ_vec, dim)
    Ω_mat_i, Ω_mat_j = generate_basis_matrices(Ω_vec, dim)

    result = diagm(j_squared(J) + s_squared(S) .- 2.0 * Σ_vec .* Ω_vec)

    mask_minus = @. (Ω_mat_i == Ω_mat_j - 1.0) & (Σ_mat_i == Σ_mat_j - 1.0)
    mask_plus = @. (Ω_mat_i == Ω_mat_j + 1.0) & (Σ_mat_i == Σ_mat_j + 1.0)

    term_minus = -j_plus(J, Ω_mat_j) .* s_minus(S, Σ_mat_j)
    term_plus = -j_minus(J, Ω_mat_j) .* s_plus(S, Σ_mat_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    return result
end

function n_dot_s(S::Float64, J::Float64, Σ_vec::Vector{Float64}, Ω_vec::Vector{Float64}, dim::Int)
    Σ_mat_i, Σ_mat_j = generate_basis_matrices(Σ_vec, dim)
    Ω_mat_i, Ω_mat_j = generate_basis_matrices(Ω_vec, dim)

    result = diagm(Ω_vec .* Σ_vec .- s_squared(S))

    mask_minus = @. (Ω_mat_i == Ω_mat_j - 1.0) & (Σ_mat_i == Σ_mat_j - 1.0)
    mask_plus = @. (Ω_mat_i == Ω_mat_j + 1.0) & (Σ_mat_i == Σ_mat_j + 1.0)

    term_minus = 0.5 * j_plus(J, Ω_mat_j) .* s_minus(S, Σ_mat_j)
    term_plus = 0.5 * j_minus(J, Ω_mat_j) .* s_plus(S, Σ_mat_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    return result
end

function jpsp_plus_jmsm(
    S::Float64,
    J::Float64,
    Λ_vec::Vector{Int},
    Σ_vec::Vector{Float64},
    Ω_vec::Vector{Float64},
    dim::Int,
)
    Λ_mat_i, Λ_mat_j = generate_basis_matrices(Λ_vec, dim)
    Σ_mat_i, Σ_mat_j = generate_basis_matrices(Σ_vec, dim)
    Ω_mat_i, Ω_mat_j = generate_basis_matrices(Ω_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (
        (Λ_mat_i == Λ_mat_j - 2) & (Ω_mat_i == Ω_mat_j - 1.0) & (Σ_mat_i == Σ_mat_j + 1.0)
    )
    mask_minus = @. (
        (Λ_mat_i == Λ_mat_j + 2) & (Ω_mat_i == Ω_mat_j + 1.0) & (Σ_mat_i == Σ_mat_j - 1.0)
    )

    term_plus = j_plus(J, Ω_mat_j) .* s_plus(S, Σ_mat_j)
    term_minus = j_minus(J, Ω_mat_j) .* s_minus(S, Σ_mat_j)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end

function sp2_plus_sm2(S::Float64, Λ_vec::Vector{Int}, Σ_vec::Vector{Float64}, dim::Int)
    Λ_mat_i, Λ_mat_j = generate_basis_matrices(Λ_vec, dim)
    Σ_mat_i, Σ_mat_j = generate_basis_matrices(Σ_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (Λ_mat_i == Λ_mat_j - 2) & (Σ_mat_i == Σ_mat_j + 2.0)
    mask_minus = @. (Λ_mat_i == Λ_mat_j + 2) & (Σ_mat_i == Σ_mat_j - 2.0)

    term_plus = s_plus(S, Σ_mat_j) .* s_plus(S, Σ_mat_j .+ 1.0)
    term_minus = s_minus(S, Σ_mat_j) .* s_minus(S, Σ_mat_j .- 1.0)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end

function jp2_plus_jm2(J::Float64, Λ_vec::Vector{Int}, Ω_vec::Vector{Float64}, dim::Int)
    Λ_mat_i, Λ_mat_j = generate_basis_matrices(Λ_vec, dim)
    Ω_mat_i, Ω_mat_j = generate_basis_matrices(Ω_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (Λ_mat_i == Λ_mat_j - 2) & (Ω_mat_i == Ω_mat_j - 2.0)
    mask_minus = @. (Λ_mat_i == Λ_mat_j + 2) & (Ω_mat_i == Ω_mat_j + 2.0)

    term_plus = j_plus(J, Ω_mat_j) .* j_plus(J, Ω_mat_j .- 1.0)
    term_minus = j_minus(J, Ω_mat_j) .* j_minus(J, Ω_mat_j .+ 1.0)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end
