include(joinpath(@__DIR__, "..", "src", "ConformalYukawaMCT.jl"))
using Printf

println("\n========== TASK 2 — ξ=0.0 baseline ==========\n")

# k-grid diagnostics
k_grid = collect(range(0.1, 8.0, length=80))
S_k_diag = yukawa_structure_factor(k_grid, 50.0, 1.0)
k_peak_idx = argmax(S_k_diag)
println("S(k) diagnostic (computed before solve, just to pick sample indices):")
@printf("   k_min  = %.4f   S = %.4f   (idx=1)\n", k_grid[1], S_k_diag[1])
@printf("   k_peak = %.4f   S = %.4f   (idx=%d)\n", k_grid[k_peak_idx], S_k_diag[k_peak_idx], k_peak_idx)
@printf("   k_max  = %.4f   S = %.4f   (idx=%d)\n", k_grid[end], S_k_diag[end], length(k_grid))

println("\n--- solve_conformal_mct(Γ=50, κ=1, ξ=0, m_sq=1, λ=1, n=1) ---")
t_start = time()
sol, kernel = solve_conformal_mct(Γ=50.0, κ=1.0, ξ=0.0, m_sq=1.0, λ=1.0, n=1.0)
t_elapsed = time() - t_start
@printf("\nTotal runtime: %.2f s\n", t_elapsed)

t_arr = get_t(sol)         # Vector{Float64}
F_vec = get_F(sol)         # Vector{Vector{Float64}}, length N_t, each length N_k
N_t = length(F_vec)
N_k = length(F_vec[1])
@printf("\nReturned solution: %d time steps × %d k-modes\n", N_t, N_k)
@printf("t range: [%.3e, %.3e]\n", t_arr[1], t_arr[end])

# convert to a dense matrix for slicing convenience: F_mat[ti, ki]
F_mat = Matrix{Float64}(undef, N_t, N_k)
for ti in 1:N_t
    F_mat[ti, :] = F_vec[ti]
end

# numerical-health scan
n_nan = count(isnan, F_mat)
n_inf = count(isinf, F_mat)
@printf("\nNaN count in F: %d\n", n_nan)
@printf("Inf count in F: %d\n", n_inf)
@printf("max |F|: %.6e\n", maximum(abs, F_mat))
@printf("min F over (t,k):  %.6e\n", minimum(F_mat))
@printf("max F over (t,k):  %.6e\n", maximum(F_mat))

# initial-condition check vs S(k)
ic_diff = maximum(abs.(F_mat[1, :] .- kernel.S_k))
@printf("max |F(t=0) - S(k)|: %.3e   (should be ≈ 0 by construction)\n", ic_diff)

# Long-time normalized correlator φ(k) = F(k, t_end) / S(k)
phi_long = F_mat[end, :] ./ kernel.S_k
@printf("\nNormalized long-time φ(k, t_end) = F(k, t_end)/S(k):\n")
@printf("   min over k:  %.4e\n", minimum(phi_long))
@printf("   max over k:  %.4e\n", maximum(phi_long))
@printf("   mean over k: %.4e\n", sum(phi_long)/length(phi_long))

# Plateau check: how much did φ change in the last decade of t?
t_mid_idx = findfirst(t -> t >= t_arr[end] / 2, t_arr)
phi_mid = F_mat[t_mid_idx, :] ./ kernel.S_k
@printf("   max |φ(t_end) - φ(t_end/2)| = %.3e\n", maximum(abs.(phi_long .- phi_mid)))
@printf("   (small ⇒ plateau reached; large ⇒ still relaxing)\n")

# Sample F(k,t) at log-spaced t for representative k
sample_idxs = [1, k_peak_idx, length(k_grid)]
labels = ["k_min", "k_peak", "k_max"]

log_targets = 10.0 .^ range(log10(max(t_arr[2], 1e-6)), log10(t_arr[end]), length=12)
sample_t_idxs = Int[1]
for τ in log_targets
    idx = findfirst(t -> t >= τ, t_arr)
    if idx !== nothing && !(idx in sample_t_idxs)
        push!(sample_t_idxs, idx)
    end
end

println("\n=== F(k,t) at representative k, log-spaced t ===")
let
    header = rpad("t", 14)
    for (lbl, i) in zip(labels, sample_idxs)
        header *= "  " * rpad("$lbl k=$(round(kernel.k_grid[i], digits=3))", 22)
    end
    println(header)
    for ti in sample_t_idxs
        row = @sprintf("%14.4e", t_arr[ti])
        for i in sample_idxs
            row *= "  " * rpad(@sprintf("%.6e", F_mat[ti, i]), 22)
        end
        println(row)
    end
end

println("\n=== F(k,t)/S(k) (normalized) at the same samples ===")
let
    header = rpad("t", 14)
    for (lbl, i) in zip(labels, sample_idxs)
        header *= "  " * rpad("$lbl S=$(round(kernel.S_k[i], digits=3))", 22)
    end
    println(header)
    for ti in sample_t_idxs
        row = @sprintf("%14.4e", t_arr[ti])
        for i in sample_idxs
            v = F_mat[ti, i] / kernel.S_k[i]
            row *= "  " * rpad(@sprintf("%.6f", v), 22)
        end
        println(row)
    end
end

# also report when each k-mode crosses 1/e of its initial value (relaxation time)
println("\n=== Relaxation time τ_α (where F(k,t)/S(k) = 1/e ≈ 0.368) ===")
for (lbl, i) in zip(labels, sample_idxs)
    phi_t = F_mat[:, i] ./ kernel.S_k[i]
    idx = findfirst(p -> p < 1/MathConstants.e, phi_t)
    if idx === nothing
        @printf("   %-7s k=%.3f : did not cross 1/e (φ_min=%.4e)\n",
                lbl, kernel.k_grid[i], minimum(phi_t))
    else
        @printf("   %-7s k=%.3f : τ_α ≈ %.4e\n", lbl, kernel.k_grid[i], t_arr[idx])
    end
end

println("\n========== END TASK 2 ==========\n")
