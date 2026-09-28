The nanobind migration still has an unsafe return path in `Crystal.GetScatteringPowerRegistry().GetObj(i)`. This surfaced while integrating the pinned branch with diffpy.srreal. It appears to be the **return-direction virtual-inheritance problem** described in the migration notes.

Reproduced on `117cb30a86dc64d0cb455cf85b53d3b269b61e65` (ObjCryst++ `30fc8df`), Linux x86_64, Python 3.14.6, nanobind 3.1.0, NumPy 2.5.1; extension built with GCC 16.2.1.

Self-contained reproducer (no diffpy.srreal import needed):

```python
from io import StringIO
from pyobjcryst.crystal import create_crystal_from_cif

crystal = create_crystal_from_cif(StringIO("""data_test
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
"""))

registry = crystal.GetScatteringPowerRegistry()
sp = registry.GetObj(0)
sp.SetBiso(0.5)
print(crystal.GetScatteringComponentList()[0].mpScattPow.GetBiso(),
      flush=True)  # prints 0.0, not the requested 0.5
print(sp.GetName(), flush=True)  # SIGSEGV
```

Running this with `python -X faulthandler` exits with SIGSEGV (subprocess return code -11). In this minimal case `SetBiso` itself returns but fails to update the actual scattering power; the subsequent `GetName` crashes. The same registry/`SetBiso` pattern has also been reported to segfault directly in downstream testing. These should not be treated as harmless shutdown leak diagnostics.

The generic registry bindings in [nb_objregistry.cpp](https://github.com/diffpy/pyobjcryst/blob/117cb30a86dc64d0cb455cf85b53d3b269b61e65/src/extensions/nb_objregistry.cpp#L60) return a base-typed `ScatteringPower&` with `reference_internal`. Unlike the explicit concrete-type handling in `nb_crystal.cpp` and `nb_scatteringcomponent.cpp`, this lets nanobind attempt the unsafe virtual-base adjustment for `ScatteringPowerAtom`.

Replacing just `sp = registry.GetObj(0)` with `sp = crystal.GetScatteringComponentList()[0].mpScattPow` yields the expected Biso of 0.5. We use that accessor as a temporary workaround in diffpy.srreal's PDF tests, keeping the original reference data and tolerances. It does not protect srfit or user scripts that use the registry pattern.

Suggested fix: apply the same explicit `dynamic_cast` to the concrete scattering-power type before `nb::cast(..., reference_internal, parent)` for registry returns, and audit the other `GetObj` overloads, indexing, and iteration for the same issue. A regression test should create scattering powers through CIF loading, access them through the registry, and verify both mutation visibility through the crystal and safe method calls.
