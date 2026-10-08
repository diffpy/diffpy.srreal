**Changed:**

* Require nanobind 3.1 or newer within the 3.x series.
* Use consistent Atom references for indexing and iteration. Surviving atoms
  follow index shifts; removed or replaced atoms detach safely. NumPy views
  retained across structural edits become snapshots; reacquire a view to
  modify the updated structure.
* Expose calculator component properties directly without preserving the
  ``PeakWidthModelOwner`` and ``ScatteringFactorTableOwner`` inheritance
  relationships. Use the calculator's own properties and methods.
* Use Python ``str`` for text parameters; decode byte strings explicitly.
  Binary serialization payloads remain ``bytes``.
* Follow nanobind's numerical conversions. Explicitly call ``float(value)``
  when an overridden ``__float__`` must be honored. ``None`` is not accepted
  as numerical input to baselines, envelopes, or scattering-factor lookup.
* Use value equality and nanobind's representation for ``QuantityType``.
  These mutable containers are unhashable, and slice assignment requires
  an iterable, including for a single replacement value.
* Interpret ``__getstate_manages_dict__`` by its truth value. Custom pickle
  hooks on guarded classes must preserve Python metadata and set this flag
  to ``True``.

**Fixed:**

* Prevent repeated Python calculation callbacks when calling base methods.
* Restore structure sequence edits and resizing contribution-array slices.
* Keep retained Atom references and NumPy views safe during structure edits
  and deserialization.
* Keep public calculator exports callable and configurable, including objects
  loaded from supported Boost.Python pickles.
* Restore saved structures and Python-defined components, honor subclass
  construction hooks, and guard against accidental metadata loss in custom
  pickle hooks.
* Accept Python and NumPy truth values in adapter anisotropy callbacks.
