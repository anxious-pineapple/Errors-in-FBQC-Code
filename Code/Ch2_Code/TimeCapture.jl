#
# Times and profiles one full capture_photon run (to t_final) at the settings planned for the sweep, and checks
# the result against the g2 eigenvalues (cavity k should end up holding nₖ). Run from the repo root, e.g.
#   julia +1.10.4 --project=. Code/Ch1RelCode/TimeCapture.jl 0.02
# (the argument is the dephasing; default 0.001). Saves, under Code/Ch1RelCode/Profiles/:
#   time_<no_cavs>cav_dep<dep*10000>.pb.gz   the profile (view: using PProf; PProf.refresh(file="..."))
#   time_<no_cavs>cav_dep<dep*10000>.h5      the MPO, sites, eigenvalues, photon numbers and timings
#

using MPOFuncs, ITensors, ITensorMPS, Profile, PProf, HDF5

const NO_CAVS, GAMMA, DT, T_FINAL = 10, 1.0, 0.01, 20.0
const TOL, MIN_DT, CUTOFF = 1e-6, 1e-12, 1e-11
const DEP = isempty(ARGS) ? 0.001 : parse(Float64, ARGS[1])
const OUT_DIR = mkpath(joinpath(@__DIR__, "Profiles"))
const NAME = "time_$(NO_CAVS)cav_dep$(round(Int, DEP * 10000))"

println("== $NO_CAVS cavities, dephasing $DEP, tol $TOL, cutoff $CUTOFF, min_dt $MIN_DT, t_final $T_FINAL")

MPOFuncs.capture_photon(2, DEP, GAMMA, DT, 0.05; tol=TOL, min_dt=MIN_DT, cutoff=CUTOFF, progress=false)  # compile

t_g2 = @elapsed couplings = MPOFuncs.photon_couplings(NO_CAVS, DEP, GAMMA, DT, T_FINAL)
println("g2: ", round(t_g2, digits=1), " s")

# coarse sampling and a big buffer, so a run of a few hours still fits (Julia samples every thread each tick)
Profile.init(n=5 * 10^7, delay=0.1)
Profile.clear()
t_evo = @elapsed (mpo, sites, eigenvals) = @profile MPOFuncs.capture_photon(NO_CAVS, DEP, GAMMA, DT, T_FINAL;
                                            tol=TOL, min_dt=MIN_DT, cutoff=CUTOFF, couplings=couplings)
pprof(; out=joinpath(OUT_DIR, "$NAME.pb.gz"), web=false)

n = real.(MPOFuncs.expect(mpo, sites))
h5open(joinpath(OUT_DIR, "$NAME.h5"), "w") do f
    write(f, "mpo", mpo); write(f, "sites", sites); write(f, "eigenvals", eigenvals)
    write(f, "photon_numbers", n); write(f, "dep", DEP); write(f, "time_g2", t_g2); write(f, "time_evolution", t_evo)
end

println("\n==== RESULT  $NO_CAVS cavities, dephasing $DEP ====")
println("time: g2 ", round(t_g2, digits=1), " s, evolution ", round(t_evo / 60, digits=1), " min   final max bond dim ",
        maxlinkdim(mpo))
println("g2 eigenvalues  ", round.(eigenvals, sigdigits=4))
println("photon numbers  ", round.(n, sigdigits=4))
println("n - eigenvalue  ", round.(n .- eigenvals, sigdigits=2))
println("total in cavities ", round(sum(n), digits=6), "   sum of eigenvalues ", round(sum(eigenvals), digits=6))
