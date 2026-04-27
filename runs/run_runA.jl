include(joinpath(@__DIR__, "..", "src", "ConformalYukawaMCT.jl"))
using Printf
using Plots

println("\n========== RUN A — ξ-sweep ==========")
println("Parameters: m²=10.0, λ=2.0, n=1.0, Γ=50.0, κ=1.0\n")

# Pre-compute n_c per ξ for context (the kernel will recompute and log it):
println("Predicted n_c = (m²/(2Γξ))^(3/4) for each ξ:")
for ξ in [0.05, 0.10, 0.15]
    nc = (10.0 / (2*50.0*ξ))^(0.75)
    @printf("   ξ=%.3f  n_c=%.4f\n", ξ, nc)
end
println("   ξ=0.000  n_c=Inf  (no conformal coupling)\n")

t0 = time()
results = xi_sweep(xi_values=[0.0, 0.05, 0.10, 0.15],
                   n=1.0, Γ=50.0, κ=1.0, m_sq=10.0, λ=2.0)
elapsed_total = time() - t0
@printf("\nTotal Run A wall time: %.2f s\n", elapsed_total)

println("\n========== RUN A — RESULTS TABLE ==========")
@printf("%6s  %8s  %14s  %14s  %10s  %12s  %14s  %10s\n",
        "ξ", "plateau", "std/max", "φ(k_min,t_end)", "F<0?", "τ_α(k_peak)", "F(k_peak,t_end)", "runtime [s]")
println("─" ^ 105)
for r in results
    plat_str = r.failed ? "FAILED" : (r.plateau ? "true" : "false")
    @printf("%6.3f  %8s  %14.4e  %14.4e  %10s  %12.4e  %14.4e  %10.2f\n",
            r.ξ, plat_str, r.ratio, r.phi_min_end,
            r.F_negative ? "yes" : "no",
            r.tau_alpha, r.F_peak_end, r.runtime)
end

println("\n=== F(k,t) shape diagnostics per ξ ===")
@printf("%6s  %12s  %12s  %12s  %18s\n",
        "ξ", "min F", "max F", "argmin k", "1/e crossed?")
for r in results
    if r.failed
        println("  failed run")
        continue
    end
    minF, minIdx = findmin(r.F_mat)
    maxF = maximum(r.F_mat)
    @printf("%6.3f  %12.4e  %12.4e  %12.3f  %18s\n",
            r.ξ, minF, maxF, r.kernel.k_grid[minIdx[2]],
            r.crossed_1e ? "yes" : "no (φ(k_peak) never hit 1/e)")
end

# Overlay plot: F(k_peak, t)/S(k_peak) per ξ
const PLOTDIR = joinpath(@__DIR__, "..", "plots"); mkpath(PLOTDIR)
plt = plot(xscale=:log10, xlabel="t", ylabel="F(k_peak,t)/S(k_peak)",
           title="ξ-sweep at n=1, Γ=50, κ=1, m²=10, λ=2",
           legend=:outertopright, size=(900, 520))
for r in results
    if r.failed
        continue
    end
    k_peak_idx = argmax(r.kernel.S_k)
    phi_peak = r.F_mat[:, k_peak_idx] ./ r.kernel.S_k[k_peak_idx]
    plot!(plt, max.(r.t_arr, 1e-7), phi_peak,
          label=@sprintf("ξ=%.3f", r.ξ), lw=1.5)
end
hline!([0.0], color=:black, lw=0.4, alpha=0.4, label="")
hline!([1/MathConstants.e], color=:gray, lw=0.4, alpha=0.4, ls=:dash, label="1/e")
savefig(plt, joinpath(PLOTDIR, "runA_F_kpeak_xi.png"))

# Also overlay F(k_min, t)/S(k_min)
plt2 = plot(xscale=:log10, xlabel="t", ylabel="F(k_min,t)/S(k_min)",
            title="ξ-sweep at k_min, n=1, Γ=50, κ=1, m²=10, λ=2",
            legend=:outertopright, size=(900, 520))
for r in results
    if r.failed
        continue
    end
    phi_min = r.F_mat[:, 1] ./ r.kernel.S_k[1]
    plot!(plt2, max.(r.t_arr, 1e-7), phi_min,
          label=@sprintf("ξ=%.3f", r.ξ), lw=1.5)
end
hline!([0.0], color=:black, lw=0.4, alpha=0.4, label="")
savefig(plt2, joinpath(PLOTDIR, "runA_F_kmin_xi.png"))

println("\nPlots written:")
println("  plots/runA_F_kpeak_xi.png")
println("  plots/runA_F_kmin_xi.png")
println("\n========== END RUN A ==========\n")
