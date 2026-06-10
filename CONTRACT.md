# Public API Contract

This document defines the stability promise `pl-row-encode` makes to its users and the
rules that keep that promise mechanically enforceable.

## The promise (SemVer)

We follow [Semantic Versioning](https://semver.org/). For a given **major** version:

- The **public surface** is frozen: the names exported from `pl_row_encode.__all__`, and
  their call signatures, do not change in a breaking way.
- The **documented behavior** of those functions does not change in a breaking way.
- The **token wire format** is stable: a token produced by any release within a major
  version decodes correctly with any later release in the same major version.

Additions (new functions, new *optional* keyword arguments, new accepted input types) are
allowed in **minor** releases. Bug fixes go in **patch** releases.

Anything that breaks one of the three guarantees above requires a **major** version bump
and an entry in the changelog.

## How it is enforced

The contract is not a gentleman's agreement — it is executable. Two layers live under
`tests/contract/`:

| File | Guards |
|------|--------|
| `test_surface.py` | The exported symbols and their exact signatures. |
| `test_behavior.py` | Documented round-trip behavior, dtypes, errors, and the token wire format (via a frozen golden token). |

### The append-only rule

Within a major version, the tests in `tests/contract/` are **append-only**:

- You **may** add new contract tests (covering new, additive API).
- You **may not** weaken, change, or delete an existing contract test.

If a change requires editing an existing contract assertion, that change is breaking by
definition. Do not edit the assertion to make CI green — instead, bump the major version
and record the break here and in the changelog.

A red contract suite means "this is a breaking change." That is the whole point.

## Dtype safety: encodable implies decodable

The core invariant is that a token we hand back can always be decoded — we never emit
bytes that only blow up on the return trip. `encode` classifies every (recursively
nested) input dtype into three tiers at encode time:

- **Known-good** — dtypes verified to round-trip losslessly (the allowlist in
  `classify_dtype`, guarded by the dtype-matrix property tests). These encode silently.
- **Known-bad** — `Categorical`. Its category→string mapping lives in an external string
  cache, not in the token, so decoding would panic. Hard-rejected at encode with an
  actionable `ComputeError` (use `Enum`, or cast to `String`).
- **Unknown** — anything we haven't vetted. We cannot safely probe it (the probe is the
  dangerous decode itself), so encode emits a `UserWarning` that the token may fail or
  panic on decode, then proceeds.

The allowlist is keyed to the **compiled** polars crate version (`Cargo.toml`), not the
user's runtime polars: the plugin decodes with its own embedded `polars-row`, so the set
of safe dtypes is fixed at build time. Bumping the polars crate is the trigger to re-run
the dtype-matrix tests and re-validate the allowlist.

## Updating the surface snapshot

`test_surface.py` pins exact signatures. When you make an **additive** change (e.g. a new
function, or a new optional keyword argument), update the snapshot in the same PR and
release it as a minor version. When you find yourself wanting to update it to *remove* or
*retype* something, that is a major change.
