include("../src/PlasmaMCTKernel.jl")
using .PlasmaMCTKernel
using Printf

const DEFAULT_DATA_FILE = joinpath(@__DIR__, "..", "data", "hnc_gamma50_kappa1.csv")

function _read_hnc_table(path::AbstractString)
    isfile(path) || error("""
    Missing HNC table: $(abspath(path))

    Provide a CSV/whitespace table with a header containing at least:
        k,S,c_short

    Optional aliases:
        K_GRID or k_grid for k
        S_K for S
        C_K_SHORT or cshort for c_short
    """)

    lines = readlines(path)
    filter!(line -> !isempty(strip(line)), lines)
    filter!(line -> !startswith(strip(line), "#"), lines)
    isempty(lines) && error("HNC table is empty: $path")

    header = split(replace(strip(lines[1]), "," => " "))
    names = Dict(lowercase(name) => i for (i, name) in enumerate(header))

    function col(name, aliases...)
        for key in (name, aliases...)
            lk = lowercase(key)
            haskey(names, lk) && return names[lk]
        end
        error("Missing column `$name` in $path")
    end

    k_col = col("k", "k_grid", "K_GRID")
    s_col = col("S", "S_k", "S_K")
    c_col = col("c_short", "C_K_SHORT", "cshort")

    rows = [parse.(Float64, split(replace(strip(line), "," => " "))) for line in lines[2:end]]
    k = [row[k_col] for row in rows]
    S = [row[s_col] for row in rows]
    c_short = [row[c_col] for row in rows]
    return k, S, c_short
end

function main(path::AbstractString=DEFAULT_DATA_FILE)
    k_grid, S_k, c_short = _read_hnc_table(path)

    K0 = plasma_mct_kernel(S_k, c_short, S_k, k_grid)
    omega2 = yukawa_bare_frequency_squared(k_grid, S_k; gamma=50, kappa=1)

    peak_index = argmax(S_k)
    high_index = length(k_grid)
    low_index = firstindex(k_grid)

    println("Gamma=50, kappa=1, xi=0 static kernel diagnostics")
    println("density = 3/(4*pi), angular nodes = 64")
    println("indices: low=$(low_index), peak=$(peak_index), high=$(high_index)")
    println()
    println("label,k,S(k),K(k,0),Omega2(k),K/Omega2")

    for (label, i) in (("k_min", low_index), ("k_peak", peak_index), ("high_k", high_index))
        ratio = omega2[i] == 0 ? NaN : K0[i] / omega2[i]
        @printf("%s,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                label, k_grid[i], S_k[i], K0[i], omega2[i], ratio)
    end

    println()
    println("This script computes the required MCT memory kernel diagnostics from the HNC table.")
    println("A full F(k,t) propagation should be run in the host MCT solver using this kernel,")
    println("because this standalone file intentionally does not add a separate time-stepper layer.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) == 0 ? DEFAULT_DATA_FILE : ARGS[1])
end
