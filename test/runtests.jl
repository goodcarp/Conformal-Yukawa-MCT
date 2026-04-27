using Test

include("../src/PlasmaMCTKernel.jl")
using .PlasmaMCTKernel

@testset "plasma_mct_kernel" begin
    k_grid = collect(range(0.0, 12.0; length=97))
    S_k = [1.0 + 1.4 * exp(-((k - 5.8) / 0.75)^2) + 0.15 * exp(-((k - 2.2) / 0.55)^2) for k in k_grid]

    # Smooth finite c_short surrogate. This is not the bare Yukawa tail; it is
    # deliberately bounded at k = 0 to exercise the WLP-regularized path.
    c_short = [-0.35 * exp(-(k / 5.0)^2) + 0.08 * exp(-((k - 5.8) / 1.1)^2) for k in k_grid]

    @testset "zero correlator gives zero memory" begin
        K = plasma_mct_kernel(zeros(length(k_grid)), c_short, S_k, k_grid; n_mu=24)
        @test all(isapprox.(K, 0.0; atol=1e-14))
    end

    @testset "positive static memory" begin
        K = plasma_mct_kernel(S_k, c_short, S_k, k_grid; n_mu=24)
        @test all(isfinite, K)
        @test all(K .>= -1e-12)
        @test maximum(K) > 0
    end

    @testset "regularized low-k limit is finite" begin
        K = plasma_mct_kernel(S_k, c_short, S_k, k_grid; n_mu=32)
        @test isapprox(K[1], 0.0; atol=1e-12)
        @test isfinite(K[2])
        @test K[2] < 1.0
    end

    @testset "collisionless beta -> 0 limit leaves only bare dispersion" begin
        c_beta0 = zeros(length(k_grid))
        S_beta0 = ones(length(k_grid))
        F0 = copy(S_beta0)
        K = plasma_mct_kernel(F0, c_beta0, S_beta0, k_grid; n_mu=16)
        @test all(isapprox.(K, 0.0; atol=1e-14))

        omega2 = yukawa_bare_frequency_squared(k_grid, S_beta0; gamma=1e12, kappa=1.0)
        F_free = collisionless_intermediate_scattering(S_beta0, omega2, 0.25)
        @test all(isfinite, F_free)
        @test isapprox(F_free[1], cos(sqrt(3.0) * 0.25); rtol=1e-14, atol=1e-14)
    end
end
