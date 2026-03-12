# Copilot instructions for SimAI

## Repository overview

- SimAI is a multi-component simulator workspace. The main components in this repository are:
  - `astra-sim-alibabacloud` for analytical/simulation/physical backends
  - `ns-3-alibabacloud` for the ns-3 network backend
  - `vidur-alibabacloud` for multi-request inference simulation support
  - `aicb` and `SimCCL` as submodules used by documented workflows
- Treat submodule-backed directories carefully and keep changes narrowly scoped to the component required by the task.

## Preferred ways to validate changes

- Follow the build commands documented in the root `README.md` and `docs/Tutorial.md`.
- Use `./scripts/build.sh -c analytical` to build the analytical binary.
- Use `./scripts/build.sh -c ns3` to build the ns-3 simulator.
- For `vidur-alibabacloud`, use its `Makefile` targets for formatting and linting:
  - `make -C vidur-alibabacloud lint`
  - `make -C vidur-alibabacloud format`
- There is no single top-level test runner documented for the whole repository, so prefer targeted validation for the component you changed.

## Change guidelines

- Keep changes minimal and avoid modifying unrelated components.
- Prefer updating existing scripts and docs instead of introducing new tooling.
- When changing user-facing workflows, keep `README.md` and `docs/Tutorial.md` consistent.
- Preserve existing command-line interfaces and example commands unless the task explicitly requires changing them.
- Do not add new dependencies unless they are necessary for the requested change.

## Environment notes

- The documented setup assumes Ubuntu 20.04, GCC/G++ 9.4.0, and Python 3.8.10.
- SimAI build documentation explicitly warns not to install `ninja` when following the source-build workflow.
