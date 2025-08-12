function j_squared(j_qn::Float64)
    j_qn * (j_qn + 1)
end

function j_plus(j_qn::Float64, omega::AbstractMatrix)
    sqrt.(j_qn * (j_qn + 1) .- omega .* (omega .- 1))
end

function j_minus(j_qn::Float64, omega::AbstractMatrix)
    sqrt.(j_qn * (j_qn + 1) .- omega .* (omega .+ 1))
end

function s_squared(s_qn::Float64)
    s_qn * (s_qn + 1)
end

function s_plus(s_qn::Float64, sigma::AbstractMatrix)
    term = s_qn * (s_qn + 1) .- sigma .* (sigma .+ 1)
    # Check for negative elements, not sure if this is the most idiomatic way of doing this
    if any(term .< 0)
        return 0.0
    end
    sqrt.(term)
end

function s_minus(s_qn::Float64, sigma::AbstractMatrix)
    term = s_qn * (s_qn + 1) .- sigma .* (sigma .- 1)
    if any(term .< 0)
        return 0.0
    end
    sqrt.(term)
end

function n_squared(s_qn::Float64, j_qn::Float64, sigma_vec::Array, omega_vec::Array, dim::Int)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = diagm(j_squared(j_qn) + s_squared(s_qn) .- 2 * sigma_vec .* omega_vec)

    mask_minus = @. (omega_i == omega_j - 1) & (sigma_i == sigma_j - 1)
    mask_plus = @. (omega_i == omega_j + 1) & (sigma_i == sigma_j + 1)

    term_minus = -j_plus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)
    term_plus = -j_minus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    result
end

function lz_sz(lambda_vec::Array, sigma_vec::Array)
    diagm(lambda_vec .* sigma_vec)
end

function three_sz2_minus_s2(s_qn::Float64, sigma_vec::Array)
    diagm(3 * sigma_vec .^ 2 .- s_squared(s_qn))
end

function n_dot_s(s_qn::Float64, j_qn::Float64, sigma_vec::Array, omega_vec::Array, dim::Int)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = diagm(omega_vec .* sigma_vec .- s_squared(s_qn))

    mask_minus = @. (omega_i == omega_j - 1) & (sigma_i == sigma_j - 1)
    mask_plus = @. (omega_i == omega_j + 1) & (sigma_i == sigma_j + 1)

    term_minus = 0.5 * j_plus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)
    term_plus = 0.5 * j_minus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)

    result[mask_minus] .= term_minus[mask_minus]
    result[mask_plus] .= term_plus[mask_plus]

    result
end

function sp2_plus_sm2(s_qn::Float64, lambda_vec::Array, sigma_vec::Array, dim::Int)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (lambda_i == lambda_j - 2) & (sigma_i == sigma_j + 2)
    mask_minus = @. (lambda_i == lambda_j + 2) & (sigma_i == sigma_j - 2)

    term_plus = s_plus(s_qn, sigma_j) .* s_plus(s_qn, sigma_j .+ 1)
    term_minus = s_minus(s_qn, sigma_j) .* s_minus(s_qn, sigma_j .- 1)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    result
end

function jpsp_plus_jmsm(s_qn::Float64, j_qn::Float64, lambda_vec::Array, sigma_vec::Array, omega_vec::Array, dim::Int)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    sigma_i, sigma_j = generate_basis_matrices(sigma_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (
        (lambda_i == lambda_j - 2) & (omega_i == omega_j - 1) & (sigma_i == sigma_j + 1)
    )
    mask_minus = @. (
        (lambda_i == lambda_j + 2) & (omega_i == omega_j + 1) & (sigma_i == sigma_j - 1)
    )

    term_plus = j_plus(j_qn, omega_j) .* s_plus(s_qn, sigma_j)
    term_minus = j_minus(j_qn, omega_j) .* s_minus(s_qn, sigma_j)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    result
end

function jp2_plus_jm2(j_qn::Float64, lambda_vec::Array, omega_vec::Array, dim::Int)
    lambda_i, lambda_j = generate_basis_matrices(lambda_vec, dim)
    omega_i, omega_j = generate_basis_matrices(omega_vec, dim)

    result = zeros(Float64, dim, dim)

    mask_plus = @. (lambda_i == lambda_j - 2) & (omega_i == omega_j - 2)
    mask_minus = @. (lambda_i == lambda_j + 2) & (omega_i == omega_j + 2)

    term_plus = j_plus(j_qn, omega_j) .* j_plus(j_qn, omega_j .- 1)
    term_minus = j_minus(j_qn, omega_j) .* j_minus(j_qn, omega_j .+ 1)

    result[mask_plus] .= term_plus[mask_plus]
    result[mask_minus] .= term_minus[mask_minus]

    result
end
