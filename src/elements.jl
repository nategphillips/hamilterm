s_squared(s_qn::Float64) = s_qn * (s_qn + 1.0)

function s_plus(s_qn::Float64, sigma::AbstractMatrix{Float64})
    term = s_qn * (s_qn + 1.0) .- sigma .* (sigma .+ 1.0)
    # Check for negative elements, not sure if this is the most idiomatic way of doing this
    if any(term .< 0.0)
        return zero(sigma)
    end

    return sqrt.(term)
end

function s_minus(s_qn::Float64, sigma::AbstractMatrix{Float64})
    term = s_qn * (s_qn + 1.0) .- sigma .* (sigma .- 1.0)
    if any(term .< 0.0)
        return zero(sigma)
    end

    return sqrt.(term)
end

j_squared(j_qn::Float64) = j_qn * (j_qn + 1.0)

function j_plus(j_qn::Float64, omega::AbstractMatrix{Float64})
    term = j_qn * (j_qn + 1.0) .- omega .* (omega .- 1.0)
    if any(term .< 0.0)
        return zero(omega)
    end

    return sqrt.(term)
end

function j_minus(j_qn::Float64, omega::AbstractMatrix{Float64})
    term = j_qn * (j_qn + 1.0) .- omega .* (omega .+ 1.0)
    if any(term .< 0.0)
        return zero(omega)
    end

    return sqrt.(term)
end

lz_sz(lambda_vec::Vector{Int}, sigma_vec::Vector{Float64}) = diagm(lambda_vec .* sigma_vec)

function three_sz2_minus_s2(s_qn::Float64, sigma_vec::Vector{Float64})
    diagm(3.0 * sigma_vec .^ 2 .- s_squared(s_qn))
end

function n_squared(
    s_qn::Float64,
    j_qn::Float64,
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    dim::Int,
)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = diagm(j_squared(j_qn) + s_squared(s_qn) .- 2.0 * sigma_vec .* omega_vec)

    mask_minus = @. (omega_i == omega_j - 1.0) & (sigma_i == sigma_j - 1.0)
    mask_plus = @. (omega_i == omega_j + 1.0) & (sigma_i == sigma_j + 1.0)

    term_minus = -j_plus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)
    term_plus = -j_minus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    return result
end

function n_dot_s(
    s_qn::Float64,
    j_qn::Float64,
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    dim::Int,
)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = diagm(omega_vec .* sigma_vec .- s_squared(s_qn))

    mask_minus = @. (omega_i == omega_j - 1.0) & (sigma_i == sigma_j - 1.0)
    mask_plus = @. (omega_i == omega_j + 1.0) & (sigma_i == sigma_j + 1.0)

    term_minus = 0.5 * j_plus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)
    term_plus = 0.5 * j_minus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    return result
end

function jpsp_plus_jmsm(
    s_qn::Float64,
    j_qn::Float64,
    lambda_vec::Vector{Int},
    sigma_vec::Vector{Float64},
    omega_vec::Vector{Float64},
    dim::Int,
)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (
        (lambda_i == lambda_j - 2) & (omega_i == omega_j - 1.0) & (sigma_i == sigma_j + 1.0)
    )
    mask_minus = @. (
        (lambda_i == lambda_j + 2) & (omega_i == omega_j + 1.0) & (sigma_i == sigma_j - 1.0)
    )

    term_plus = j_plus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)
    term_minus = j_minus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end

function sp2_plus_sm2(s_qn::Float64, lambda_vec::Vector{Int}, sigma_vec::Vector{Float64}, dim::Int)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (lambda_i == lambda_j - 2) & (sigma_i == sigma_j + 2.0)
    mask_minus = @. (lambda_i == lambda_j + 2) & (sigma_i == sigma_j - 2.0)

    term_plus = s_plus(s_qn, sigma_j) .* s_plus(s_qn, sigma_j .+ 1.0)
    term_minus = s_minus(s_qn, sigma_j) .* s_minus(s_qn, sigma_j .- 1.0)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end

function jp2_plus_jm2(j_qn::Float64, lambda_vec::Vector{Int}, omega_vec::Vector{Float64}, dim::Int)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (lambda_i == lambda_j - 2) & (omega_i == omega_j - 2.0)
    mask_minus = @. (lambda_i == lambda_j + 2) & (omega_i == omega_j + 2.0)

    term_plus = j_plus(j_qn, omega_j) .* j_plus(j_qn, omega_j .- 1.0)
    term_minus = j_minus(j_qn, omega_j) .* j_minus(j_qn, omega_j .+ 1.0)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    return result
end
