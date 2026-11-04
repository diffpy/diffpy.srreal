**Changed:**

* Structure conversion now returns ordinary atomic or periodic adapters,
  without reading ``pdffit`` metadata. Configure PDF parameters explicitly
  through the existing calculator interfaces.
* ``nometa()`` returns or converts the source adapter without a proxy.
  ``None`` returns the empty adapter.

**Deprecated:**

* ``NoMetaStructureAdapter``, retained for saved objects and subclasses.
  Its Python constructor now emits ``DeprecationWarning``. Historical
  metadata adapter classes remain importable for pickle compatibility.

**Removed:**

* The ``_customPQConfig`` bindings and Python callback dispatch, including
  automatic application of structure metadata to PDF calculators.
