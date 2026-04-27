# ConformalYukawaMCT.jl
#
# Custom memory kernel for ModeCouplingTheory.jl implementing the
# conformally-coupled Yukawa one-component plasma (OCP).
#
# Free energy:
#   f(n,φ) = f₀(n) − η n^(4/3) A²(φ) + ½m²φ² + (λ/4)φ⁴
# with A(φ) = 1 + ½ξφ², so A²(φ) = 1 + ξφ² + ¼ξ²φ⁴.
#
# Equivalently:
#   f = f₀(n) + ½ r(n) φ² + ¼ u(n) φ⁴
# with
#   r(n) = m² − 2ηξ n^(4/3)
#   u(n) = λ − ηξ² n^(4/3)
#
# Critical density: n_c = (m² / 2ηξ)^(3/4)
# Transition character at n_c: second-order if λ > ½ξm², else first-order.
#
# This file plugs into the ModeCouplingTheory.jl solver
# (Pihlajamaa et al., JOSS 2023, arXiv:2305.01365).
# Install with: Pkg.add("ModeCouplingTheory")
#
# Usage:
#   include("ConformalYukawaMCT.jl")
#   sol = solve_conformal_mct(Γ=50.0, κ=1.0, ξ=0.5, m²=1.0, λ=1.0, n=1.0)
#   plot(get_t(sol), get_F(sol))

using ModeCouplingTheory
using ModeCouplingTheory: evaluate_kernel, evaluate_kernel!
using LinearAlgebra
using SpecialFunctions
using FFTW
using Printf
using Interpolations
using Statistics
import ModeCouplingTheory: evaluate_kernel, evaluate_kernel!

# Regularized HNC structure-factor table (Γ=50, κ=1). Provides
# HNC_K_GRID, HNC_S_K, HNC_C_K (full) and HNC_C_K_SHORT (Wertheim-
# Lebowitz-Percus regularization, c_short = c_HNC − c_long, where
# c_long is the bare Yukawa already carried by Ω(k)).
# Γ-dependence is implicit; regenerate if Γ or κ change.
include(joinpath(@__DIR__, "sk_g50_regularized.jl"))

# Linear interpolation onto an arbitrary k-grid.
# Out-of-range: S → 1.0 (asymptotic), c → 0.0 (asymptotic).
function hnc_structure_factor(k_grid)
    itp = linear_interpolation(HNC_K_GRID, HNC_S_K, extrapolation_bc=1.0)
    return [itp(k) for k in k_grid]
end

# Direct correlation routed through the short-range part c_short to
# avoid double-counting the bare Yukawa already in Ω(k).
function hnc_direct_correlation(k_grid)
    itp = linear_interpolation(HNC_K_GRID, HNC_C_K_SHORT, extrapolation_bc=0.0)
    return [itp(k) for k in k_grid]
end

# ─────────────────────────────────────────────────────────────────
# 1. Static structure factor S(k) for the Yukawa OCP
# ─────────────────────────────────────────────────────────────────
# Use the Hypernetted-Chain (HNC) closure to compute S(k) and the
# direct correlation function c(k). For Yukawa OCP at strong coupling,
# we use a parameterized form fit to MD data
# (Hamaguchi-Farouki-Dubin, PRE 56, 4671, 1997).
#
# This is an approximation. A production calculation would solve the
# full HNC equations iteratively. The fit suffices for testing the
# framework's qualitative predictions.

function yukawa_structure_factor(k_grid, Γ, κ)
    S_k = similar(k_grid)
    
    # Peak position scales weakly with κ; for κ~1, k_peak ≈ 5.3 (in units of 1/a)
    k_peak = 5.3 * (1.0 + 0.1 * (κ - 1.0))
    
    # Peak height grows with Γ (sharper correlation peak in stronger coupling)
    h_peak = 1.0 + 0.04 * sqrt(Γ) + 0.5
    
    # Compressibility S(0)
    S_zero = 1.0 / (1.0 + 0.05 * Γ)
    
    σ = 0.85
    
    for (i, k) in enumerate(k_grid)
        if k < 1.0
            S_k[i] = S_zero + (1.0 - S_zero) * (k/1.0)^2 * 0.4
        else
            S_k[i] = 1.0 + (h_peak - 1.0) * exp(-((k - k_peak)/σ)^2)
        end
    end
    
    return max.(S_k, 0.04)
end

# Direct correlation function via Ornstein-Zernike: c(k) = (S(k)-1) / (n S(k))
function direct_correlation(S_k, n_density)
    return (S_k .- 1.0) ./ (n_density .* S_k)
end

# ─────────────────────────────────────────────────────────────────
# 2. The conformally-coupled memory kernel
# ─────────────────────────────────────────────────────────────────
# The standard one-component MCT memory kernel is
#   M(k,t) = (n / 2) ∫ dq/(2π)³ |V(k,q)|² F(q,t) F(|k-q|,t)
# where the vertex is
#   V(k,q) = (k̂·q) c(q) + (k̂·(k-q)) c(|k-q|)
#
# For the conformally-coupled system, the effective ion charge is
# modified by A(φ), so the vertex picks up a multiplicative factor
# that depends on the scalar field's vacuum expectation value.
#
# In the symmetric phase (n < n_c, ⟨φ⟩=0), the leading correction
# is from scalar fluctuations: ⟨A²⟩ = 1 + ξ⟨φ²⟩.
# The fluctuation-fluctuation propagator gives ⟨φ²⟩ = T/r(n) at mean field,
# so the effective coupling diverges as n → n_c.
#
# In the broken phase (n > n_c, ⟨φ⟩=φ₀), the vertex picks up the
# nonzero VEV: A²(φ₀) = 1 + ξφ₀² + ¼ξ²φ₀⁴.

struct ConformalYukawaKernel{T} <: ModeCouplingTheory.MemoryKernel
    k_grid::Vector{T}      # wavevector grid
    S_k::Vector{T}         # static structure factor  
    c_k::Vector{T}         # direct correlation function
    V_sq::Matrix{T}        # squared vertex matrix (precomputed)
    n_density::T           # ion number density
    Γ::T                   # Coulomb coupling parameter
    κ::T                   # screening parameter
    ξ::T                   # conformal coupling
    m_sq::T                # bare scalar mass squared
    λ::T                   # quartic self-coupling
    A_sq_eff::T            # effective ⟨A²⟩ in current phase
    in_broken_phase::Bool  # whether we're above n_c
    φ_vev::T               # scalar VEV (zero in symmetric phase)
end

function compute_vertex_matrix(k_grid, c_k, n_density)
    N = length(k_grid)
    dk = k_grid[2] - k_grid[1]
    V_sq = zeros(N, N)
    
    for i in 1:N, j in 1:N
        k = k_grid[i]
        q = k_grid[j]
        # Angle-averaged: use representative |k-q|
        p = sqrt(k^2 + q^2)  # simplified; full would integrate over angle
        p_idx = clamp(round(Int, (p - k_grid[1])/dk) + 1, 1, N)
        
        cq = c_k[j]
        cp = c_k[p_idx]
        
        # Vertex magnitude
        V_kq = (q/max(k, 0.1)) * cq + (p/max(k, 0.1)) * cp
        V_sq[i,j] = V_kq^2
    end
    
    return V_sq
end

function ConformalYukawaKernel(; Γ::T, κ::T, ξ::T, m_sq::T, λ::T,
                                 n::T, k_max::T = 8.0, N_k::Int = 80) where T
    # Critical density and transition diagnostics
    η = Γ  # in our reduced units; would be A_OCP × Z²e²/(4πε₀) × (4π/3)^(1/3) physically
    n_c = (m_sq / (2η * ξ))^(0.75)
    transition_2nd_order = λ > 0.5 * ξ * m_sq
    
    @info "ConformalYukawa kernel initialization" Γ κ ξ m_sq λ n n_c transition_2nd_order
    
    # k-grid
    k_grid = collect(range(0.1, k_max, length=N_k))
    
    # Static structure
    S_k = hnc_structure_factor(k_grid)
    n_density = n
    c_k = hnc_direct_correlation(k_grid)
    
    # Determine phase
    in_broken_phase = n > n_c
    
    # Compute VEV in broken phase: φ₀² = -r(n)/u(n) where r,u are coefficients
    r_n = m_sq - 2η * ξ * n^(4/3)
    u_n = λ - η * ξ^2 * n^(4/3)
    
    if in_broken_phase && u_n > 0
        φ_vev = sqrt(-r_n / u_n)
    elseif in_broken_phase && u_n <= 0
        @warn "Quartic coefficient negative — first-order regime, framework breaks down here"
        φ_vev = zero(T)
    else
        φ_vev = zero(T)
    end
    
    # Effective A² in the current phase
    if in_broken_phase
        A_sq_eff = 1 + ξ * φ_vev^2 + 0.25 * ξ^2 * φ_vev^4
    else
        # Symmetric phase: include leading fluctuation correction
        # ⟨φ²⟩ = T / r(n) at mean field; T = 1/Γ in reduced units
        T_temp = 1.0 / Γ
        if r_n > 1e-6
            phi_sq_avg = T_temp / r_n
            A_sq_eff = 1 + ξ * phi_sq_avg
        else
            A_sq_eff = 1e6  # divergent near critical point
        end
    end
    
    # Vertex modified by conformal factor
    V_sq = compute_vertex_matrix(k_grid, c_k, n_density) .* A_sq_eff
    
    return ConformalYukawaKernel(k_grid, S_k, c_k, V_sq, n_density,
                                  Γ, κ, ξ, m_sq, λ, A_sq_eff,
                                  in_broken_phase, φ_vev)
end

# Evaluate kernel into a Diagonal matrix, as ModeCouplingTheory.jl expects.
# M(k,t) = (n/2) Σ_q V_sq[k,q] F[q,t]² dk (simplified bilinear)
# In full MCT: F(q,t) F(|k-q|,t), but we use the angle-averaged simplification
function evaluate_kernel!(out::Diagonal, kernel::ConformalYukawaKernel, F, t)
    N = length(kernel.k_grid)
    dk = kernel.k_grid[2] - kernel.k_grid[1]
    @inbounds for i in 1:N
        s = zero(eltype(F))
        for j in 1:N
            s += kernel.V_sq[i,j] * F[j]^2
        end
        out.diag[i] = 0.5 * kernel.n_density * s * dk
    end
    return out
end

function evaluate_kernel(kernel::ConformalYukawaKernel, F, t)
    out = Diagonal(similar(F, length(kernel.k_grid)))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

# ─────────────────────────────────────────────────────────────────
# 3. The MCT equation and solution driver
# ─────────────────────────────────────────────────────────────────

function setup_conformal_mct(; Γ::Float64=50.0, κ::Float64=1.0,
                                ξ::Float64=0.5, m_sq::Float64=1.0,
                                λ::Float64=1.0, n::Float64=1.0,
                                k_max::Float64=8.0, N_k::Int=80,
                                t_max::Float64=1e6)
    
    kernel = ConformalYukawaKernel(; Γ, κ, ξ, m_sq, λ, n, k_max, N_k)
    
    N = length(kernel.k_grid)
    
    # Bare frequency: Ω²(k) = k² T / S(k) + ω_p²/(1+(k/κ)²)
    T_temp = 1.0 / Γ
    ωp_sq_yukawa = 1.0 ./ (1.0 .+ (kernel.k_grid ./ κ).^2)
    Ω_sq = kernel.k_grid.^2 .* T_temp ./ kernel.S_k .+ ωp_sq_yukawa
    
    # MCT equation: F̈(k,t) + Ω²(k) F(k,t) + ∫₀ᵗ K(t-τ) Ḟ(k,τ) dτ = 0
    # In ModeCouplingTheory.jl form: α=1, β=0, γ=Ω², δ=0, kernel as above
    α = ones(N)
    β = zeros(N)
    γ = Ω_sq
    δ = zeros(N)
    
    F_initial = kernel.S_k        # F(k, 0) = S(k)
    F_dot_initial = zeros(N)      # Ḟ(k, 0) = 0
    
    equation = MemoryEquation(α, β, γ, δ, F_initial, F_dot_initial, kernel)
    
    return equation, kernel
end

function solve_conformal_mct(; kwargs...)
    equation, kernel = setup_conformal_mct(; kwargs...)

    t_max = Float64(get(kwargs, :t_max, 1e6))
    @info "Solving conformally-coupled MCT equation..." t_max
    solver = TimeDoublingSolver(t_max=t_max, verbose=false)
    sol = solve(equation, solver)

    return sol, kernel
end

# ─────────────────────────────────────────────────────────────────
# 4. Diagnostic: extract the dynamic structure factor
# ─────────────────────────────────────────────────────────────────

function compute_dynamic_structure_factor(sol, kernel; n_omega=1024)
    # NOTE: get_F returns Vector{Vector{Float64}} with non-uniform t (doubling
    # grid). The FFT here treats dt = t_arr[end]/N_t as if the samples were
    # uniformly spaced — they are not. Resulting ω axis is therefore a
    # numerical placeholder, not a calibrated frequency. Reported as-is.
    t_arr = get_t(sol)
    F_vec = get_F(sol)
    N_t = length(F_vec)
    N_k = length(F_vec[1])
    F_arr = Matrix{Float64}(undef, N_t, N_k)
    for ti in 1:N_t
        F_arr[ti, :] = F_vec[ti]
    end
    dt = t_arr[end] / N_t

    # Window
    window = [0.5 - 0.5*cos(2π*i/(N_t-1)) for i in 0:N_t-1]

    # Normalize: φ(k,t) = F(k,t) / S(k)
    phi = F_arr ./ kernel.S_k'

    # FFT each k-mode (zero-pad if N_t < n_omega; truncate if longer).
    if N_t > n_omega
        n_omega = nextpow(2, N_t)
    end
    S_kw = zeros(n_omega ÷ 2 + 1, N_k)
    for k_idx in 1:N_k
        signal = phi[:, k_idx] .* window
        padded = vcat(signal, zeros(n_omega - N_t))
        spec = abs.(fft(padded))[1:n_omega÷2+1]
        S_kw[:, k_idx] = spec * kernel.S_k[k_idx]
    end

    omega = collect(0:n_omega÷2) .* (2π / (n_omega * dt))

    return omega, S_kw
end

function find_low_k_modes(sol, kernel)
    omega, S_kw = compute_dynamic_structure_factor(sol, kernel)

    println("\n─── Low-k mode analysis ───")
    println("Looking for propagating scalar mode at k → 0")
    println("$(rpad("k", 8)) $(rpad("S(k)", 8)) $(rpad("ω_peak", 10)) $(rpad("Q", 8)) character")
    println("─" ^ 50)

    rows = NamedTuple[]
    for k_idx in [1, 3, 5, 8, 12, 20, 40]
        if k_idx > length(kernel.k_grid); continue; end

        spec = S_kw[:, k_idx]
        # Skip DC bins
        peak_idx = argmax(spec[5:end]) + 4
        ω_peak = omega[peak_idx]
        amp = spec[peak_idx]

        # Q-factor estimate
        half = amp / 2
        L = peak_idx; R = peak_idx
        while L > 5 && spec[L] > half; L -= 1; end
        while R < length(spec) && spec[R] > half; R += 1; end
        width = max(omega[R] - omega[L], 0.01)
        Q = ω_peak / width

        char = if ω_peak < 0.05
            "diffusive"
        elseif Q > 3
            "PROPAGATING"
        elseif Q > 1
            "damped"
        else
            "overdamped"
        end

        @printf("%8.3f %8.3f %10.3f %8.2f %s\n",
                kernel.k_grid[k_idx], kernel.S_k[k_idx], ω_peak, Q, char)

        push!(rows, (k_idx=k_idx, k=kernel.k_grid[k_idx],
                     Sk=kernel.S_k[k_idx], ω_peak=ω_peak, Q=Q,
                     character=char))
    end
    return rows
end

# ─────────────────────────────────────────────────────────────────
# 5. Parameter sweep: scan n through n_c, look for mode softening
# ─────────────────────────────────────────────────────────────────

function xi_sweep(; xi_values=[0.0, 0.2, 0.5, 1.0], n=1.0,
                    Γ=50.0, κ=1.0, m_sq=1.0, λ=1.0)
    results = NamedTuple[]
    for ξ in xi_values
        println("\n━━━ ξ = $ξ ━━━")
        t_start = time()
        local sol, kernel
        try
            sol, kernel = solve_conformal_mct(; Γ, κ, ξ, m_sq, λ, n)
        catch err
            elapsed = time() - t_start
            @warn "solver failed" ξ exception=(err, catch_backtrace())
            push!(results, (ξ=ξ, plateau=false, ratio=NaN,
                            phi_min_end=NaN, F_negative=false,
                            tau_alpha=NaN, F_peak_end=NaN,
                            crossed_1e=false, runtime=elapsed,
                            failed=true, F_mat=nothing,
                            t_arr=nothing, kernel=nothing))
            continue
        end
        elapsed = time() - t_start

        t_arr = get_t(sol)
        F_vec = get_F(sol)
        N_t = length(F_vec)
        N_k = length(kernel.k_grid)
        F_mat = Matrix{Float64}(undef, N_t, N_k)
        for ti in 1:N_t
            F_mat[ti, :] = F_vec[ti]
        end

        k_peak_idx = argmax(kernel.S_k)
        k_min_idx = 1
        phi_peak = F_mat[:, k_peak_idx] ./ kernel.S_k[k_peak_idx]
        crossing = findfirst(p -> p < 1/MathConstants.e, phi_peak)
        crossed_1e = crossing !== nothing
        τ_α = crossed_1e ? t_arr[crossing] : t_arr[end]

        # Plateau metric: std/max in window [τ_α/10, τ_α/3].
        # If τ_α never crossed 1/e, sample [t_end/10, t_end/3] instead.
        window_lo = τ_α / 10
        window_hi = τ_α / 3
        in_window = findall(t -> window_lo <= t <= window_hi, t_arr)
        if length(in_window) >= 3
            window_F = F_mat[in_window, k_peak_idx]
            std_F = std(window_F)
            max_amp = maximum(F_mat[:, k_peak_idx])
            ratio = max_amp > 0 ? std_F / max_amp : NaN
            plateau = isfinite(ratio) && ratio < 0.1
        else
            std_F = NaN
            ratio = NaN
            plateau = false
        end

        phi_min_end = F_mat[end, k_min_idx] / kernel.S_k[k_min_idx]
        F_peak_end = F_mat[end, k_peak_idx]
        F_negative = any(F_mat .< 0)

        push!(results, (ξ=ξ, plateau=plateau, ratio=ratio,
                        phi_min_end=phi_min_end, F_negative=F_negative,
                        tau_alpha=τ_α, F_peak_end=F_peak_end,
                        crossed_1e=crossed_1e, runtime=elapsed,
                        failed=false, F_mat=F_mat, t_arr=t_arr,
                        kernel=kernel))
    end
    return results
end

function critical_density_sweep(; Γ=50.0, κ=1.0, ξ=0.5, m_sq=1.0, λ=1.0)
    n_c = (m_sq / (2*Γ*ξ))^(0.75)
    println("Predicted critical density: n_c = $n_c")
    println("Transition character: ", λ > 0.5*ξ*m_sq ? "SECOND-ORDER" : "FIRST-ORDER")

    n_values = [0.3*n_c, 0.6*n_c, 0.85*n_c, 0.95*n_c, 1.05*n_c, 1.2*n_c]

    results = NamedTuple[]
    for n in n_values
        println("\n━━━ n/n_c = $(n/n_c) ━━━")
        t_start = time()
        local sol, kernel
        local converged = true
        local err_msg = ""
        try
            sol, kernel = solve_conformal_mct(; Γ, κ, ξ, m_sq, λ, n)
        catch err
            converged = false
            err_msg = sprint(showerror, err)
            elapsed = time() - t_start
            @warn "solver failed at n=$n" exception=(err, catch_backtrace())
            push!(results, (n=n, n_over_nc=n/n_c, converged=false,
                            err_msg=err_msg, runtime=elapsed,
                            phi_min_end=NaN, F_peak_end=NaN,
                            F_min_reached_zero=false, n_negative_modes=-1,
                            min_F=NaN, low_k_rows=nothing,
                            F_mat=nothing, t_arr=nothing, kernel=nothing))
            continue
        end
        elapsed = time() - t_start

        t_arr = get_t(sol)
        F_vec = get_F(sol)
        N_t = length(F_vec)
        N_k = length(kernel.k_grid)
        F_mat = Matrix{Float64}(undef, N_t, N_k)
        for ti in 1:N_t
            F_mat[ti, :] = F_vec[ti]
        end

        n_nan = count(isnan, F_mat)
        n_inf = count(isinf, F_mat)
        if n_nan > 0 || n_inf > 0
            converged = false
            err_msg = "NaN=$n_nan, Inf=$n_inf"
        end

        k_peak_idx = argmax(kernel.S_k)
        phi_min_end = F_mat[end, 1] / kernel.S_k[1]
        F_peak_end = F_mat[end, k_peak_idx]
        F_min_reached_zero = abs(F_mat[end, 1] / kernel.S_k[1]) < 1e-3
        n_negative_modes = count(k -> minimum(F_mat[:, k]) < 0, 1:N_k)
        min_F_val = minimum(F_mat)

        # find_low_k_modes prints AND returns the data (we capture it).
        low_k_rows = find_low_k_modes(sol, kernel)

        push!(results, (n=n, n_over_nc=n/n_c, converged=converged,
                        err_msg=err_msg, runtime=elapsed,
                        phi_min_end=phi_min_end, F_peak_end=F_peak_end,
                        F_min_reached_zero=F_min_reached_zero,
                        n_negative_modes=n_negative_modes,
                        min_F=min_F_val, low_k_rows=low_k_rows,
                        F_mat=F_mat, t_arr=t_arr, kernel=kernel))
    end

    return results
end

# ─────────────────────────────────────────────────────────────────
# Entry point for testing
# ─────────────────────────────────────────────────────────────────

# Run with:
#   julia> include("ConformalYukawaMCT.jl")
#   julia> sol, kernel = solve_conformal_mct(Γ=50.0, ξ=0.5, m_sq=1.0, λ=1.0, n=0.5)
#   julia> find_low_k_modes(sol, kernel)
#
# Or sweep through the critical density:
#   julia> results = critical_density_sweep(Γ=50.0, ξ=0.5, m_sq=1.0, λ=1.0)
