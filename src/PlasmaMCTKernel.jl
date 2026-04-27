module PlasmaMCTKernel

using LinearAlgebra

export plasma_mct_kernel,
       yukawa_bare_frequency_squared,
       collisionless_intermediate_scattering

const DEFAULT_DENSITY = 3 / (4pi)
const DEFAULT_ANGULAR_NODES = 64

"""
    plasma_mct_kernel(F::Vector, c_short::Vector, S_k::Vector, k_grid::Vector; kwargs...)::Vector

Evaluate the regularized 3D Yukawa-OCP mode-coupling memory kernel

    K(k,t) = n/(16*pi^3) * integral d^3q V(k,q)^2 F(q,t)F(|k-q|,t)

with the full isotropic angular reduction

    K(k,t) = n/(8*pi^2) * integral dq q^2 integral_-1^1 dmu
              [q*mu*c_short(q) + (k - q*mu)*c_short(p)]^2 F(q,t)F(p,t),
    p = sqrt(k^2 + q^2 - 2*k*q*mu).

The vertex uses the supplied `c_short` only. For a Yukawa plasma this should be
the Wertheim-Lebowitz-Percus regularized direct correlation function, i.e. the
HNC direct correlation with the bare long-range Yukawa Fourier component
removed. `S_k` is validated for shape/finite input and kept in the public
signature because it is usually available alongside `F`, but the unnormalized
kernel above uses the supplied `F` directly.

Keyword arguments:
- `density = 3/(4*pi)` for reduced OCP units with Wigner-Seitz radius `a = 1`.
- `n_mu = 64` Gauss-Legendre nodes for the angular integral.

Numerical assumptions:
- `k_grid` is a strictly increasing, uniformly spaced radial grid.
- The radial `q` integral is truncated to the supplied grid and integrated by
  trapezoidal weights.
- Values at `p < first(k_grid)` are clamped to the first tabulated value. Values
  at `p > last(k_grid)` are set to zero, so modes outside the HNC table do not
  contribute spuriously.
"""
function plasma_mct_kernel(
    F::Vector,
    c_short::Vector,
    S_k::Vector,
    k_grid::Vector;
    density::Real = DEFAULT_DENSITY,
    n_mu::Integer = DEFAULT_ANGULAR_NODES,
)::Vector{Float64}
    _validate_inputs(F, c_short, S_k, k_grid, n_mu)

    k = Float64.(k_grid)
    f = Float64.(F)
    c = Float64.(c_short)
    q_weights = _trapezoid_weights(k)
    mu_nodes, mu_weights = _gausslegendre(Int(n_mu))

    n_k = length(k)
    kernel = zeros(Float64, n_k)
    prefactor = Float64(density) / (8pi^2)

    @inbounds for ik in 1:n_k
        kval = k[ik]
        if iszero(kval)
            # The longitudinal vertex cancels exactly at k = 0.
            kernel[ik] = 0.0
            continue
        end

        accum = 0.0
        for iq in 1:n_k
            q = k[iq]
            fq = f[iq]
            iszero(fq) && continue

            cq = c[iq]
            q_weighted_volume = q^2 * q_weights[iq]
            angular_sum = 0.0

            for imu in eachindex(mu_nodes)
                mu = mu_nodes[imu]
                p2 = kval^2 + q^2 - 2 * kval * q * mu
                p = sqrt(max(p2, 0.0))

                fp = _interp_grid(k, f, p)
                iszero(fp) && continue

                cp = _interp_grid(k, c, p)
                vertex = q * mu * cq + (kval - q * mu) * cp
                angular_sum += mu_weights[imu] * vertex^2 * fq * fp
            end

            accum += q_weighted_volume * angular_sum
        end

        kernel[ik] = prefactor * accum
    end

    return kernel
end

"""
    yukawa_bare_frequency_squared(k_grid, S_k; gamma=50, kappa=1, density=3/(4*pi), mass=1)

Bare longitudinal frequency for the reduced Yukawa OCP,

    Omega^2(k) = k^2*kBT/(m*S(k)) + omega_p^2/(1 + (k/kappa)^2),
    kBT = 1/gamma, omega_p^2 = 4*pi*n/m.
"""
function yukawa_bare_frequency_squared(
    k_grid::AbstractVector,
    S_k::AbstractVector;
    gamma::Real = 50,
    kappa::Real = 1,
    density::Real = DEFAULT_DENSITY,
    mass::Real = 1,
)::Vector{Float64}
    length(k_grid) == length(S_k) || throw(ArgumentError("k_grid and S_k must have the same length"))
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
    kappa > 0 || throw(ArgumentError("kappa must be positive"))
    mass > 0 || throw(ArgumentError("mass must be positive"))

    kBT = 1 / Float64(gamma)
    omega_p2 = 4pi * Float64(density) / Float64(mass)

    omega2 = Vector{Float64}(undef, length(k_grid))
    @inbounds for i in eachindex(k_grid, S_k)
        s = Float64(S_k[i])
        s > 0 || throw(ArgumentError("S_k must be positive for the bare frequency"))
        kval = Float64(k_grid[i])
        omega2[i] = kval^2 * kBT / (Float64(mass) * s) + omega_p2 / (1 + (kval / Float64(kappa))^2)
    end
    return omega2
end

"""
    collisionless_intermediate_scattering(S_k, omega2, t)

Free Newtonian density correlator used for the beta -> 0 regression test. In
that limit `c_short -> 0`, the MCT memory kernel vanishes and the dynamics are
the bare oscillator `F(k,t) = S(k)cos(Omega(k)t)`.
"""
function collisionless_intermediate_scattering(
    S_k::AbstractVector,
    omega2::AbstractVector,
    t::Real,
)::Vector{Float64}
    length(S_k) == length(omega2) || throw(ArgumentError("S_k and omega2 must have the same length"))
    return [Float64(S_k[i]) * cos(sqrt(max(Float64(omega2[i]), 0.0)) * Float64(t)) for i in eachindex(S_k)]
end

function _validate_inputs(F, c_short, S_k, k_grid, n_mu)
    n = length(k_grid)
    n >= 3 || throw(ArgumentError("k_grid must contain at least three points"))
    length(F) == n || throw(ArgumentError("F and k_grid must have the same length"))
    length(c_short) == n || throw(ArgumentError("c_short and k_grid must have the same length"))
    length(S_k) == n || throw(ArgumentError("S_k and k_grid must have the same length"))
    n_mu >= 2 || throw(ArgumentError("n_mu must be at least 2"))

    all(isfinite, F) || throw(ArgumentError("F must contain only finite values"))
    all(isfinite, c_short) || throw(ArgumentError("c_short must contain only finite values"))
    all(isfinite, S_k) || throw(ArgumentError("S_k must contain only finite values"))
    all(isfinite, k_grid) || throw(ArgumentError("k_grid must contain only finite values"))

    k = Float64.(k_grid)
    all(diff(k) .> 0) || throw(ArgumentError("k_grid must be strictly increasing"))
    dk = k[2] - k[1]
    atol = max(1e-12, 1e-10 * abs(dk))
    all(abs((k[i + 1] - k[i]) - dk) <= atol for i in 1:(n - 1)) ||
        throw(ArgumentError("k_grid must be uniformly spaced"))

    return nothing
end

function _trapezoid_weights(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    weights = Vector{Float64}(undef, n)
    weights[1] = (x[2] - x[1]) / 2
    @inbounds for i in 2:(n - 1)
        weights[i] = (x[i + 1] - x[i - 1]) / 2
    end
    weights[n] = (x[n] - x[n - 1]) / 2
    return weights
end

function _gausslegendre(n::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    beta = [i / sqrt(4i^2 - 1) for i in 1:(n - 1)]
    jacobi = SymTridiagonal(zeros(n), beta)
    eig = eigen(jacobi)
    nodes = Vector{Float64}(eig.values)
    weights = vec(2 .* eig.vectors[1, :].^2)
    return nodes, weights
end

@inline function _interp_grid(x::Vector{Float64}, y::Vector{Float64}, xi::Float64)::Float64
    xi <= x[1] && return y[1]
    xi > x[end] && return 0.0

    dx = x[2] - x[1]
    i = floor(Int, (xi - x[1]) / dx) + 1
    i >= length(x) && return y[end]

    t = (xi - x[i]) / dx
    return (1 - t) * y[i] + t * y[i + 1]
end

end # module

