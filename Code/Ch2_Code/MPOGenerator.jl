#
# Sweeps dephasing for a fixed number of cavities, saving the capture_photon MPO for every dephasing
# value plus a CSV of the g2 eigenvalues (photon numbers per virtual cavity) across the sweep.
# Run twice: once with the evolver's cash_karppe step size floored at min_dt = 1e-4 ("coarse", trades
# accuracy near t=0 for speed), once with no floor ("fine", capture_photon's own unfloored default).
# g2 (the couplings) doesn't depend on those settings, so it is computed once per dephasing value and
# shared when both runs are done together.
#
# Output, under a coarse/ and fine/ subfolder of this file's own directory:
#   MPOData/<tag>/mpo_dep<dep*10000>_no_cavs<no_cavs>.h5   HDF5 groups "mpo", "sites"; datasets "eigenvals", "dep"
#   MPOData/<tag>/eigenvalues_no_cavs<no_cavs>.csv         one row per dep, columns dep, eigenval_1..eigenval_<no_cavs>
#

using ITensors
using ITensorMPS
using MPOFuncs
using HDF5
using DelimitedFiles

# ----------------------------------------------------------------------------------------------
# Sweep parameters
# ----------------------------------------------------------------------------------------------
const GAMMA = 1.0
const NO_CAVS = 10
const DT = 0.01
const T_FINAL = 20.0
const DEP_LIST = 0:0.001:0.02
const CUTOFF = 1e-9     # truncation cutoff for ρ and everything inside drho
const DATA_DIR = joinpath(@__DIR__, "MPOData")

# dep -> integer code used in filenames (0.0005 -> 5, 0.02 -> 200), matching the dep*10000 convention
# already used for the peps_dep*.jld2 files in Data/.
depcode(dep) = round(Int, dep * 10000)

mpo_filename(tag_dir, dep, no_cavs) =
    joinpath(tag_dir, "mpo_dep$(depcode(dep))_no_cavs$(no_cavs).h5")

eigval_csv_filename(tag_dir, no_cavs) =
    joinpath(tag_dir, "eigenvalues_no_cavs$(no_cavs).csv")

# ----------------------------------------------------------------------------------------------
# Dephasing sweep for one or more runs, given as tag => min_dt pairs (min_dt is capture_photon's
# cash_karppe step floor; 1e-12 reproduces the unfloored default). For each dep value the couplings
# (g2) are computed once and shared by all the runs, then each run saves its MPO (HDF5) under
# data_dir/tag/. Each run's eigenvalue CSV is written at the end. Returns tag => data matrix.
# ----------------------------------------------------------------------------------------------
function run_dephasing_sweep(runs; no_cavs=NO_CAVS, gamma=GAMMA, dt=DT, t_final=T_FINAL,
                              dep_list=DEP_LIST, data_dir=DATA_DIR, cutoff=CUTOFF)
    tag_dirs = Dict(tag => mkpath(joinpath(data_dir, tag)) for (tag, _) in runs)
    data = Dict(tag => Matrix{Float64}(undef, length(dep_list), no_cavs + 1) for (tag, _) in runs)

    for (row, dep) in enumerate(dep_list)
        println("dephasing = $dep: computing couplings (g2)")
        couplings = MPOFuncs.photon_couplings(no_cavs, dep, gamma, dt, t_final)

        for (tag, min_dt) in runs
            println("[$tag] dephasing = $dep (min_dt = $min_dt, cutoff = $cutoff)")

            mpo, sites, eigenvals = MPOFuncs.capture_photon(no_cavs, dep, gamma, dt, t_final; min_dt=min_dt,
                                                            cutoff=cutoff, couplings=couplings)

            h5open(mpo_filename(tag_dirs[tag], dep, no_cavs), "w") do f
                write(f, "mpo", mpo)
                write(f, "sites", sites)
                write(f, "eigenvals", eigenvals)
                write(f, "dep", dep)
            end

            data[tag][row, 1] = dep
            data[tag][row, 2:end] = eigenvals
        end
    end

    header = ["dep"; ["eigenval_$i" for i in 1:no_cavs]]
    for (tag, _) in runs
        open(eigval_csv_filename(tag_dirs[tag], no_cavs), "w") do io
            writedlm(io, permutedims(header), ',')
            writedlm(io, data[tag], ',')
        end
    end
    return data
end

# ----------------------------------------------------------------------------------------------
# Only runs the actual sweeps when this file is executed directly, not when it is `include`-d.
#   julia +1.10.4 --project=. Code/Ch1RelCode/MPOGenerator.jl coarse   -> coarse only
#   julia +1.10.4 --project=. Code/Ch1RelCode/MPOGenerator.jl fine     -> fine only
#   julia +1.10.4 --project=. Code/Ch1RelCode/MPOGenerator.jl          -> both, sharing g2 per dephasing
#                                                                         (coarse then fine at each dep value)
# ----------------------------------------------------------------------------------------------
const RUNS = Dict("coarse" => 1e-4, "fine" => 1e-12)

if abspath(PROGRAM_FILE) == @__FILE__
    selected = isempty(ARGS) ? ["coarse", "fine"] : ARGS
    for tag in selected
        haskey(RUNS, tag) || error("unknown run '$tag', expected one of: $(join(keys(RUNS), ", "))")
    end
    run_dephasing_sweep([tag => RUNS[tag] for tag in selected])
end
