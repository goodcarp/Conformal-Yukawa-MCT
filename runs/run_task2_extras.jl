include(joinpath(@__DIR__, "..", "src", "ConformalYukawaMCT.jl"))
using Printf
using Plots

sol, kernel = solve_conformal_mct(Γ=50.0, κ=1.0, ξ=0.0, m_sq=1.0, λ=1.0, n=1.0)
t_arr = get_t(sol)
F_vec = get_F(sol)
N_t = length(F_vec)
N_k = length(F_vec[1])
F_mat = Matrix{Float64}(undef, N_t, N_k)
for ti in 1:N_t
    F_mat[ti, :] = F_vec[ti]
end

# Locate the most-negative entry and the first-crossing-zero per mode
min_idx = argmin(F_mat)
@printf("\nMost-negative F entry: F[t=%.4e, k=%.3f] = %.6e (S(k)=%.4f)\n",
        t_arr[min_idx[1]], kernel.k_grid[min_idx[2]],
        F_mat[min_idx[1], min_idx[2]], kernel.S_k[min_idx[2]])

# Per-k: does F ever go below zero? if so, when, and what's the minimum?
println("\nPer-k negative excursion summary (only k where F goes negative):")
@printf("%6s  %8s  %12s  %12s  %12s\n", "k_idx", "k", "min_F", "t_at_min", "first_zero_t")
for k_idx in 1:N_k
    col = F_mat[:, k_idx]
    if minimum(col) < 0
        ti_min = argmin(col)
        # first zero crossing from the IC side
        first_neg = findfirst(<(0.0), col)
        t_first_neg = first_neg === nothing ? NaN : t_arr[first_neg]
        @printf("%6d  %8.3f  %12.4e  %12.4e  %12.4e\n",
                k_idx, kernel.k_grid[k_idx], minimum(col),
                t_arr[ti_min], t_first_neg)
    end
end

# Identify which range of k goes negative
neg_modes = [k_idx for k_idx in 1:N_k if minimum(F_mat[:, k_idx]) < 0]
@printf("\nNumber of k modes with F<0 at some t: %d / %d\n", length(neg_modes), N_k)
if !isempty(neg_modes)
    @printf("k range with negative excursion: [%.3f, %.3f]\n",
            kernel.k_grid[neg_modes[1]], kernel.k_grid[neg_modes[end]])
end

# Generate plot: F(k,t) vs t for several k, log-x, normalized
const PLOTDIR = joinpath(@__DIR__, "..", "plots"); mkpath(PLOTDIR)
plot_idxs = [1, 10, 30, 53, 60, 80]  # low, mid, peak, post-peak, high
plt = plot(xscale=:log10, xlabel="t", ylabel="F(k,t)/S(k)",
           title="ξ=0, Γ=50, κ=1, n=1 baseline",
           legend=:outertopright, size=(800, 500))
for ki in plot_idxs
    plot!(plt, max.(t_arr, 1e-7), F_mat[:, ki] ./ kernel.S_k[ki],
          label=@sprintf("k=%.3f, S=%.3f", kernel.k_grid[ki], kernel.S_k[ki]),
          lw=1.5)
end
hline!([0.0], color=:black, lw=0.5, alpha=0.4, label="")
savefig(plt, joinpath(PLOTDIR, "task2_F_normalized.png"))

plt2 = plot(xscale=:log10, xlabel="t", ylabel="F(k,t)",
            title="ξ=0, Γ=50, κ=1, n=1 baseline (raw F)",
            legend=:outertopright, size=(800, 500))
for ki in plot_idxs
    plot!(plt2, max.(t_arr, 1e-7), F_mat[:, ki],
          label=@sprintf("k=%.3f, S=%.3f", kernel.k_grid[ki], kernel.S_k[ki]),
          lw=1.5)
end
hline!([0.0], color=:black, lw=0.5, alpha=0.4, label="")
savefig(plt2, joinpath(PLOTDIR, "task2_F_raw.png"))

println("\nPlots written to plots/task2_F_normalized.png and plots/task2_F_raw.png")
