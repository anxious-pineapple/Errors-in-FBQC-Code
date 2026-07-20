# Errors in FBQC Code
(package installing pecularities explained)


## What this repo is

A Julia project (`Project.toml` / `Manifest.toml`) that also runs some plain
Python notebooks. Julia's package manager (via `CondaPkg.jl`, using `pixi`
under the hood) automatically manages an isolated Python environment for
Python-interop packages (`PythonPlot`, `PyPlot`, etc.) and for any extra
Python packages declared in `CondaPkg.toml`.

You don't install or manage conda/pixi yourself — Julia drives it.

## Setup on a new machine

```
juliaup add 1.10.4        # match the Julia version Manifest.toml was resolved with
julia +1.10.4 --project=. -e 'using Pkg; Pkg.instantiate()'
```

This downloads all Julia packages and, as a side effect of building
`PythonPlot`/`CondaPkg`, resolves the Python environment too (including
anything declared in the root `CondaPkg.toml`).

If `Pkg.instantiate()` complains about `Manifest.toml` being out of sync with
`Project.toml` (e.g. "X is a direct dependency but does not appear in the
manifest"), someone edited `Project.toml` without re-resolving — see
"Adding dependencies" below for the right fix.

## Daily workflow

**Julia** (scripts, REPL, Julia notebooks):
```
julia +1.10.4 --project=.
```
- `+1.10.4` pins the Julia version — using a different version can force an
  unwanted re-resolve of `Manifest.toml`.
- `--project=.` uses this repo's environment instead of your global one.
- `]` enters Pkg mode for `add`/`rm`/`status`/`instantiate`.

Julia notebooks in VS Code: pick the "Julia 1.10.4" kernel (registered
automatically by IJulia after instantiation).

**Python notebooks** (e.g. `Code/StimFBQC.ipynb`): pick the interpreter at
`.CondaPkg/.pixi/envs/default/bin/python3` as the kernel. This env is
Julia-managed — see below before `pip install`ing into it directly.

## Where dependencies live

| File | What it's for | Edit by hand? |
|---|---|---|
| `Project.toml` | Direct Julia dependencies | Via `Pkg.add`/`Pkg.rm`, not by hand |
| `Manifest.toml` | Exact resolved versions (lockfile) | No — Julia writes this |
| `CondaPkg.toml` (root) | Extra Python packages your code needs that no Julia package already pulls in | Yes |
| `.CondaPkg/` | Auto-generated Python env + pixi lockfile | Never — gitignored, delete and regenerate if broken |

Every Julia package that needs Python ships its own `CondaPkg.toml`
declaring what it needs. `CondaPkg.jl` merges all of these (including the
root one) and resolves one shared Python env from the union.

## Adding dependencies

**New Julia package**:
```
julia --project=.
] add PackageName
```
This updates `Project.toml` and `Manifest.toml` together — commit both in
the same commit. Never let them drift across separate commits.

**New Python package your notebooks need** (not something a Julia package
already pulls in): add it to the root `CondaPkg.toml`'s `[pip.deps]` table
(or `[deps]` if it's a conda package), then run:
```
julia --project=. -e 'using CondaPkg; CondaPkg.resolve()'
```
to actually install it. Commit `CondaPkg.toml` — never commit `.CondaPkg/`
itself (it's gitignored on purpose, it's a build artifact).
