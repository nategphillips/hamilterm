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
    eta::Float64 = 0.0
end

@kwdef struct SpinSpinConsts
    lambda::Float64 = 0.0
    lambda_D::Float64 = 0.0
    lambda_H::Float64 = 0.0
    theta::Float64 = 0.0
end

@kwdef struct SpinRotationConsts
    gamma::Float64 = 0.0
    gamma_D::Float64 = 0.0
    gamma_H::Float64 = 0.0
    gamma_L::Float64 = 0.0
    gamma_S::Float64 = 0.0
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

@kwdef struct AllConsts
    rotational::RotationalConsts = RotationalConsts()
    spin_orbit::SpinOrbitConsts = SpinOrbitConsts()
    spin_spin::SpinSpinConsts = SpinSpinConsts()
    spin_rotation::SpinRotationConsts = SpinRotationConsts()
    lambda_doubling::LambdaDoublingConsts = LambdaDoublingConsts()
end
