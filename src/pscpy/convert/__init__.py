"""Converters between psc output format versions.

There is one module per format update, each converting from one version to
the next (e.g. ``legacy_to_v1``, later ``v1_to_v2``). To bring old output up
to date, run the converters in sequence::

    python -m pscpy.convert.legacy_to_v1 -o OUTDIR SRC.bp ...
"""
