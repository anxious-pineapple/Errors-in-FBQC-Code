#
# Profiles capture_photon's time evolution with PProf, using the same settings as MPOGenerator.jl but
# stopping early at T_STOP. The couplings (g2) are computed beforehand for the full T_FINAL window, so the
# dynamics are the real ones but g2 isn't in the profile. Run from the repo root:
#   julia +1.10.4 --project=. Code/Ch1RelCode/ProfileCapture.jl
# or include it from the VS Code REPL. Opens the PProf web UI (http://localhost:57599) when done, and
# saves the profile to Code/Ch1RelCode/capture_photon_profile.pb.gz.
# View a saved profile again later with:  using PProf; PProf.refresh(file="<path>.pb.gz")
#

using MPOFuncs
using Profile
using PProf

const NO_CAVS = 5
const DEP = 0.001
const GAMMA = 1.0
const DT = 0.01
const T_FINAL = 20.0
const T_STOP = 0.02
const MIN_DT = 1e-4
const CUTOFF = 1e-9
const OUT = joinpath(@__DIR__, "capture_photon_profile.pb.gz")

# compile everything first, so the profile isn't dominated by JIT compilation
MPOFuncs.capture_photon(2, DEP, GAMMA, DT, 0.05; min_dt=MIN_DT, cutoff=CUTOFF, progress=false)

println("Computing couplings (g2), not profiled...")
@time couplings = MPOFuncs.photon_couplings(NO_CAVS, DEP, GAMMA, DT, T_FINAL)

# 10 ms sampling with room for ~10^7 frames: enough for 10+ minutes of deep ITensors stacks
Profile.init(n=10^7, delay=0.01)
Profile.clear()
@time @profile MPOFuncs.capture_photon(NO_CAVS, DEP, GAMMA, DT, T_FINAL; min_dt=MIN_DT, cutoff=CUTOFF,
                                        t_stop=T_STOP, couplings=couplings)

pprof(; out=OUT, web=true)
println("Profile saved to $OUT, web UI at http://localhost:57599")

if !isinteractive()
    println("Press Enter to close the web UI and exit.")
    readline()
end
