# Notice and Attribution

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).

## Third-Party Code and Attributions

### QuantLib-Risks-Cpp (Xcelerit Computing Ltd.)

This project contains code derived from or inspired by:

- **Repository**: https://github.com/auto-differentiation/QuantLib-Risks-Cpp
- **Copyright**: (C) 2010-2024 Xcelerit Computing Ltd.
- **License**: AGPL-3.0-or-later

Files derived from QuantLib-Risks-Cpp retain their original copyright notices.

### QuantLib

This project integrates with QuantLib:

- **Repository**: https://github.com/lballabio/QuantLib
- **License**: QuantLib License (BSD-3-Clause style, permissive)

QuantLib itself is not modified or included in this repository. Integration is achieved through:
1. A patch file applied at build time (`patches/quantlib-abool.patch`)
2. Header-only adapter code

### Forge

This project depends on Forge for JIT compilation and automatic differentiation:

- **Repository**: https://github.com/da-roth/forge
- **License**: zlib (permissive)

Forge is consumed as a pre-built package via `find_package(Forge)`.

---

## Future Relicensing

The intent is to eventually rewrite AGPL-derived portions clean-room and relicense
this project under a permissive license (MIT/BSD/zlib). Until then, this project
remains AGPL-3.0-or-later.
