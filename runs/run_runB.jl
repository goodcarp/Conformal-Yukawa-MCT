include(joinpath(@__DIR__, "..", "src", "ConformalYukawaMCT.jl"))
using Printf
using Plots

println("\n========== RUN B — n-sweep at ξ=0.1 ==========")
println("Parameters: Γ=50.0, κ=1.0, ξ=0.1, m²=10.0, λ=2.0\n")

t0 = time()
results = critical_density_sweep(Γ=50.0, κ=1.0, ξ=0.1, m_sq=10.0, λ=2.0)
elapsed_total = time() - t0
@printf("\n\nTotal Run B wall time: %.2f s\n", elapsed_total)

println("\n========== RUN B — RESULTS TABLE ==========")
@printf("%-8s  %-9s  %-12s  %-15s  %-15s  %-12s  %-7s  %-12s  %-10s\n",
        "n", "n/n_c", "converged", "φ(k_min,t_end)", "F(k_peak,t_end)",
        "min F", "F<0?", "F_min→0?", "runtime[s]")
println("─" ^ 130)
for r in results
    nstr = @sprintf("%.4f", r.n)
    ncstr = @sprintf("%.3f", r.n_over_nc)
    convstr = r.converged ? "yes" : "FAIL: $(r.err_msg)"
    if r.F_mat === nothing
        @printf("%-8s  %-9s  %-12s  %s\n", nstr, ncstr, convstr, "—")
    else
        @printf("%-8s  %-9s  %-12s  %-15.4e  %-15.4e  %-12.4e  %-7s  %-12s  %-10.2f\n",
                nstr, ncstr, convstr,
                r.phi_min_end, r.F_peak_end, r.min_F,
                r.n_negative_modes > 0 ? "yes($(r.n_negative_modes))" : "no",
                r.F_min_reached_zero ? "yes" : "no", r.runtime)
    end
end

println("\n========== find_low_k_modes() per density ==========")
println("(NB: ω_peak comes from FFT of F(k,t) on a NON-uniform doubling time grid;")
println("  the package returns log-spaced t. The FFT routine in compute_dynamic_structure_factor")
println("  treats dt as t_end/N_t, i.e. assumes uniform spacing it does not have. The numbers")
println("  below are what the routine outputs; their interpretation as physical frequencies is")
println("  compromised by that mismatch. Reporting them as requested.)\n")

@printf("%-8s  %-8s  %-8s  %-10s  %-8s  %-10s  %s\n",
        "n/n_c", "k_idx", "k", "S(k)", "ω_peak", "Q", "character")
println("─" ^ 78)
for r in results
    if r.low_k_rows === nothing; continue; end
    for row in r.low_k_rows
        @printf("%-8.3f  %-8d  %-8.3f  %-10.4f  %-8.3f  %-10.3f  %s\n",
                r.n_over_nc, row.k_idx, row.k, row.Sk,
                row.ω_peak, row.Q, row.character)
    end
    println()
end

# Diagnostic: are ω_peak values constant across k?
println("\n=== ω_peak constancy check (flag for FFT-on-non-uniform-grid artifact) ===")
@printf("%-8s  %-12s  %-12s  %-12s\n", "n/n_c", "ω_peak min", "ω_peak max", "max-min")
for r in results
    if r.low_k_rows === nothing; continue; end
    ωs = [row.ω_peak for row in r.low_k_rows]
    @printf("%-8.3f  %-12.4e  %-12.4e  %-12.4e\n",
            r.n_over_nc, minimum(ωs), maximum(ωs), maximum(ωs)-minimum(ωs))
end

# Plot F(k_peak, t)/S(k_peak) overlay for all densities
const PLOTDIR = joinpath(@__DIR__, "..", "plots"); mkpath(PLOTDIR)
plt = plot(xscale=:log10, xlabel="t", ylabel="F(k_peak,t)/S(k_peak)",
           title="n-sweep at ξ=0.1, Γ=50, κ=1, m²=10, λ=2",
           legend=:outertopright, size=(900, 520))
for r in results
    if r.F_mat === nothing; continue; end
    k_peak_idx = argmax(r.kernel.S_k)
    phi_peak = r.F_mat[:, k_peak_idx] ./ r.kernel.S_k[k_peak_idx]
    plot!(plt, max.(r.t_arr, 1e-7), phi_peak,
          label=@sprintf("n/n_c=%.2f", r.n_over_nc), lw=1.5)
end
hline!([0.0], color=:black, lw=0.4, alpha=0.4, label="")
hline!([1/MathConstants.e], color=:gray, lw=0.4, alpha=0.4, ls=:dash, label="1/e")
savefig(plt, joinpath(PLOTDIR, "runB_F_kpeak_n.png"))

plt2 = plot(xscale=:log10, xlabel="t", ylabel="F(k_min,t)/S(k_min)",
            title="n-sweep at k_min, ξ=0.1, Γ=50, κ=1, m²=10, λ=2",
            legend=:outertopright, size=(900, 520))
for r in results
    if r.F_mat === nothing; continue; end
    phi_min = r.F_mat[:, 1] ./ r.kernel.S_k[1]
    plot!(plt2, max.(r.t_arr, 1e-7), phi_min,
          label=@sprintf("n/n_c=%.2f", r.n_over_nc), lw=1.5)
end
hline!([0.0], color=:black, lw=0.4, alpha=0.4, label="")
savefig(plt2, joinpath(PLOTDIR, "runB_F_kmin_n.png"))

println("\nPlots written:")
println("  plots/runB_F_kpeak_n.png")
println("  plots/runB_F_kmin_n.png")
println("\n========== END RUN B ==========\n")
