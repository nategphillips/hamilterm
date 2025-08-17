@kwdef struct RotationalConsts
    B::Float64 = 0.0
    D::Float64 = 0.0
    H::Float64 = 0.0
    L::Float64 = 0.0
    M::Float64 = 0.0
    P::Float64 = 0.0
end

@kwdef struct SpinOrbitConsts
    A::Float64 = 0.0
    A_D::Float64 = 0.0
    A_H::Float64 = 0.0
    A_L::Float64 = 0.0
    A_M::Float64 = 0.0
    η::Float64 = 0.0
end

@kwdef struct SpinSpinConsts
    λ::Float64 = 0.0
    λ_D::Float64 = 0.0
    λ_H::Float64 = 0.0
    θ::Float64 = 0.0
end

@kwdef struct SpinRotationConsts
    γ::Float64 = 0.0
    γ_D::Float64 = 0.0
    γ_H::Float64 = 0.0
    γ_L::Float64 = 0.0
    γ_S::Float64 = 0.0
end

@kwdef struct LambdaDoublingConsts
    p::Float64 = 0.0
    o::Float64 = 0.0
    q::Float64 = 0.0
    o_D::Float64 = 0.0
    p_D::Float64 = 0.0
    q_D::Float64 = 0.0
    o_H::Float64 = 0.0
    p_H::Float64 = 0.0
    q_H::Float64 = 0.0
    o_L::Float64 = 0.0
    p_L::Float64 = 0.0
    q_L::Float64 = 0.0
end

@kwdef struct Consts
    rotational::RotationalConsts = RotationalConsts()
    spin_orbit::SpinOrbitConsts = SpinOrbitConsts()
    spin_spin::SpinSpinConsts = SpinSpinConsts()
    spin_rotation::SpinRotationConsts = SpinRotationConsts()
    lambda_doubling::LambdaDoublingConsts = LambdaDoublingConsts()
end
