#
# Compares step-control settings (tol, min_dt) for capture_photon on one case, with a PProf profile of each.
# Runs to t_final, where cavity k should hold exactly the g2 eigenvalue nₖ, and compares against those (and
# against a tight-tolerance reference run, which separates stepper error from error already in the couplings).
# Reports time, step statistics (incl. steps accepted at the min_dt floor above tolerance), final bond dimension
# and photon number per cavity. Run from the repo root:
#   julia +1.10.4 --project=. Code/Ch1RelCode/CompareStepControl.jl
# Profiles go to Code/Ch1RelCode/Profiles/<name>.pb.gz; view one with
#   using PProf; PProf.refresh(file="Code/Ch1RelCode/Profiles/<name>.pb.gz")
#

using MPOFuncs, ITensors, ITensorMPS, Profile, PProf
using LinearAlgebra: norm

const NO_CAVS, DEP, GAMMA, DT, T_FINAL, CUTOFF = 5, 0.001, 1.0, 0.01, 20.0, 1e-9
const PROFILE_DIR = mkpath(joinpath(@__DIR__, "Profiles"))

# name, tol, min_dt, cutoff, profile?
# 2×2 of tol and cutoff, no floor (an earlier run with tol 1e-4 / 1e-3 and floor 1e-4 never hit the floor here).
# The last one is the reference that `max |n - ref|` is measured against.
const VARIANTS = [
    ("tol1e-4_cut1e-9",  1e-4, 1e-12, 1e-9,  true),     # current "fine"
    ("tol1e-4_cut1e-11", 1e-4, 1e-12, 1e-11, true),
    ("tol1e-6_cut1e-9",  1e-6, 1e-12, 1e-9,  true),
    ("tol1e-6_cut1e-11", 1e-6, 1e-12, 1e-11, true),     # reference
]

# cash_karppe as in MPOFuncs, plus step counters
const STATS = Dict{Symbol,Any}()
reset_stats!() = merge!(STATS, Dict(:accepted => 0, :rejected => 0, :floor => 0, :floor_worst => 0.0,
                                    :dt_min => Inf))
Core.eval(MPOFuncs, quote
    function cash_karppe(sys, d_rho, t, dt; tol=1e-4, p_=nothing, max_dt=Inf, safety=0.9, min_dt=1e-12)
        S = Main.STATS
        cut_off = p_[5]
        weighted(w, ks) = [w[i] * ks[i] for i in eachindex(w) if w[i] != 0]
        dt = max(dt, min_dt)
        while true
            k_s = k_gen(sys, d_rho, t, dt, CK_a, CK_b, p=p_)
            err = norm(+(weighted(CK_cꜛ .- CK_c, k_s)...; cutoff=cut_off))
            if err <= tol || dt <= min_dt
                S[:accepted] += 1
                S[:dt_min] = min(S[:dt_min], dt)
                if err > tol
                    S[:floor] += 1
                    S[:floor_worst] = max(S[:floor_worst], err / tol)
                end
                dt_next = min(dt * min(5.0, safety * (tol/max(err, eps(tol)))^0.2), max_dt)
                return +(sys, weighted(CK_c, k_s)...; cutoff=cut_off), dt, dt_next
            end
            S[:rejected] += 1
            dt = max(dt * max(0.1, safety * (tol/err)^0.25), min_dt)
        end
    end
end)

photon_numbers(mpo, sites) = real.(MPOFuncs.expect(mpo, sites))

# probability that every cavity is empty: the escaped photon (modes beyond NO_CAVS) should show up here
function vacuum_probability(mpo, sites)
    v = ITensor(1.0)
    for k in eachindex(sites)
        v *= mpo[k] * op("Ground", sites[k])
    end
    return real(scalar(v)) / real(MPOFuncs.trace(mpo, sites))
end

reset_stats!()
MPOFuncs.capture_photon(2, DEP, GAMMA, DT, 0.05; min_dt=1e-4, cutoff=CUTOFF, progress=false)   # compile
println("Computing couplings (g2)...")
couplings = MPOFuncs.photon_couplings(NO_CAVS, DEP, GAMMA, DT, T_FINAL)

results = []
for (name, tol, min_dt, cut, prof) in VARIANTS
    println("\n== $name (tol = $tol, min_dt = $min_dt, cutoff = $cut)")
    reset_stats!()
    run() = MPOFuncs.capture_photon(NO_CAVS, DEP, GAMMA, DT, T_FINAL; tol=tol, min_dt=min_dt, cutoff=cut,
                                    couplings=couplings)
    if prof
        Profile.init(n=10^7, delay=0.01); Profile.clear()
        t = @elapsed (m, s, _) = @profile run()
        pprof(; out=joinpath(PROFILE_DIR, "$name.pb.gz"), web=false)
    else
        t = @elapsed (m, s, _) = run()
    end
    push!(results, (name, t, copy(STATS), maxlinkdim(m), photon_numbers(m, s), vacuum_probability(m, s)))
end

ideal = couplings[2]     # g2 eigenvalues: cavity k should end up holding nₖ
ref = results[end][5]
println("\n==== $NO_CAVS cavities, dephasing $DEP, t = 0 -> $T_FINAL ====")
println(rpad("g2 eigenvalues", 19), " n = ", round.(ideal, sigdigits=6), "  total ", round(sum(ideal), digits=6),
        "  (escaped beyond cavity $NO_CAVS ≈ ", round(1 - exp(-GAMMA * T_FINAL) - sum(ideal), sigdigits=3), ")")
for (name, t, S, χ, n, p0) in results
    println(rpad(name, 19), " time ", lpad(round(t, digits=1), 6), " s | steps ", lpad(S[:accepted], 5),
            "  rejected ", lpad(S[:rejected], 4), "  at floor over tol ", lpad(S[:floor], 4),
            " (worst ", round(S[:floor_worst], digits=1), "× tol)  smallest dt ", round(S[:dt_min], sigdigits=2),
            " | χ ", χ)
    println(" "^19, " n = ", round.(n, sigdigits=6), "  total ", round(sum(n), digits=6),
            "  P(all cavities empty) = ", round(p0, sigdigits=3))
    println(" "^19, " n - eigenvalue = ", round.(n .- ideal, sigdigits=2),
            "   max |n - ref| ", round(maximum(abs.(n .- ref)), sigdigits=2))
end
