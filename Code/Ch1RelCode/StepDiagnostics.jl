#
# Step-by-step diagnostics of capture_photon's time stepping for a few (tol, cutoff) settings: logs every step
# attempt (t, dt, error estimate, accepted?, bond dimension of ρ and of the stage derivatives k, time taken) and
# compares the final photon numbers with the g2 eigenvalues. Run from the repo root:
#   julia +1.10.4 --project=. Code/Ch1RelCode/StepDiagnostics.jl
#

using MPOFuncs, ITensors, ITensorMPS, Statistics
using LinearAlgebra: norm

const NO_CAVS, DEP, GAMMA, DT, T_FINAL = 5, 0.001, 1.0, 0.01, 20.0
const VARIANTS = [("tol1e-4 cut1e-9 ", 1e-4, 1e-9), ("tol1e-4 cut1e-11", 1e-4, 1e-11),
                  ("tol1e-6 cut1e-9 ", 1e-6, 1e-9), ("tol1e-6 cut1e-11", 1e-6, 1e-11)]

# cash_karppe as in MPOFuncs, logging every attempt to LOG
const LOG = NamedTuple[]
Core.eval(MPOFuncs, quote
    function cash_karppe(sys, d_rho, t, dt; tol=1e-4, p_=nothing, max_dt=Inf, safety=0.9, min_dt=1e-12)
        cut_off = p_[5]
        weighted(w, ks) = [w[i] * ks[i] for i in eachindex(w) if w[i] != 0]
        dt = max(dt, min_dt)
        χ_in = maxlinkdim(sys)
        while true
            t0 = time_ns()
            k_s = k_gen(sys, d_rho, t, dt, CK_a, CK_b, p=p_)
            χ_k = maximum(maxlinkdim, k_s)
            err = norm(+(weighted(CK_cꜛ .- CK_c, k_s)...; cutoff=cut_off))
            ok = err <= tol || dt <= min_dt
            out = ok ? +(sys, weighted(CK_c, k_s)...; cutoff=cut_off) : nothing
            push!(Main.LOG, (t=t, dt=dt, err=err, ok=ok, χ_in=χ_in, χ_k=χ_k, secs=(time_ns() - t0) / 1e9))
            if ok
                dt_next = min(dt * min(5.0, safety * (tol/max(err, eps(tol)))^0.2), max_dt)
                return out, dt, dt_next
            end
            dt = max(dt * max(0.1, safety * (tol/err)^0.25), min_dt)
        end
    end
end)

MPOFuncs.capture_photon(2, DEP, GAMMA, DT, 0.05; cutoff=1e-9, progress=false)   # compile
couplings = MPOFuncs.photon_couplings(NO_CAVS, DEP, GAMMA, DT, T_FINAL)
ideal = couplings[2]

χ_at(L, tq) = (i = findfirst(e -> e.t >= tq, L); i === nothing ? L[end].χ_in : L[i].χ_in)
println("\n==== $NO_CAVS cavities, dephasing $DEP, t = 0 -> $T_FINAL ====")
println("g2 eigenvalues   ", round.(ideal, sigdigits=4))
println("'noise ×': when a rejected step is retried smaller, how much less the error estimate shrank than (dt ratio)^5")
println("           (median over retries; ≈1 = real step error, ≫1 = truncation noise)\n")
for (name, tol, cut) in VARIANTS
    empty!(LOG)
    secs = @elapsed (mpo, sites, _) = MPOFuncs.capture_photon(NO_CAVS, DEP, GAMMA, DT, T_FINAL; tol=tol, cutoff=cut,
                                                             couplings=couplings, progress=false)
    n = real.(MPOFuncs.expect(mpo, sites))
    L = copy(LOG); acc = filter(e -> e.ok, L)
    retries = [(L[i+1].err / L[i].err) / (L[i+1].dt / L[i].dt)^5 for i in 1:length(L)-1 if !L[i].ok && L[i+1].t == L[i].t]
    early = filter(e -> e.t < 1.0, L)
    println(name, " | ", lpad(round(secs, digits=1), 6), " s  steps ", lpad(length(acc), 5), "  rejected ",
            lpad(length(L) - length(acc), 4), "  noise × ", isempty(retries) ? "–" : round(median(retries), sigdigits=2))
    println(" "^16, " | ρ bond dim at t = 0.01, 0.1, 1, 5, 20: ", [χ_at(acc, x) for x in (0.01, 0.1, 1.0, 5.0, 20.0)],
            "  max ", maximum(e -> e.χ_in, acc), "   stage k's max ", maximum(e -> e.χ_k, L))
    println(" "^16, " | s/step ", round(mean(e -> e.secs, L), sigdigits=2), "   attempts with t < 1: ", length(early),
            " (", round(sum(e -> e.secs, early), digits=1), " s)   accepted err/tol median ",
            round(median([e.err / tol for e in acc]), sigdigits=2), "   smallest dt ", round(minimum(e -> e.dt, acc), sigdigits=2))
    println(" "^16, " | n - eigenvalue = ", round.(n .- ideal, sigdigits=2))
end
