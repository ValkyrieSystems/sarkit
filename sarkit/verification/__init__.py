"""
#########################################
Verification (:mod:`sarkit.verification`)
#########################################

Verification of SAR data in NGA standard formats.

Consistency Checking
====================

Python Interface
----------------

.. autosummary::
   :toctree: generated/
   :recursive:

   CphdConsistency
   CrsdConsistency
   SicdConsistency
   SiddConsistency

Consistency objects should be instantiated using ``from_parts`` when data components are available in memory or
``from_file`` when data has already been serialized into a standard format.

.. doctest::

   >>> import sarkit.verification as skver

   >>> with open("data/example-cphd-1.0.1.xml", "r") as f:
   ...     con = skver.CphdConsistency.from_file(f)
   >>> con.check()
   >>> bool(con.passes())
   True
   >>> bool(con.failures())
   False

   >>> import lxml.etree
   >>> cphd_xmltree = lxml.etree.parse("data/example-cphd-1.0.1.xml")
   >>> con = skver.CphdConsistency.from_parts(cphd_xmltree)
   >>> con.check()
   >>> bool(con.passes())
   True
   >>> bool(con.failures())
   False

Command-Line Interface
----------------------

Each of the consistency checkers has a corresponding entry point:

.. code-block:: shell-session

   $ cphdcheck /path/to/file
   $ crsdcheck /path/to/file
   $ sicdcheck /path/to/file
   $ siddcheck /path/to/file

The command line flags for each are given below:

.. _cphdcheck-cli:

.. autoprogram:: sarkit.verification._cphdcheck:_parser()
   :prog: cphdcheck

.. _crsdcheck-cli:

.. autoprogram:: sarkit.verification._crsdcheck:_parser()
   :prog: crsdcheck

.. _sicdcheck-cli:

.. autoprogram:: sarkit.verification._sicdcheck:_parser()
   :prog: sicdcheck

.. _siddcheck-cli:

.. autoprogram:: sarkit.verification._siddcheck:_parser()
   :prog: siddcheck
"""

from ._cphd_consistency import (
    CphdConsistency,
)
from ._crsd_consistency import (
    CrsdConsistency,
)
from ._sicd_consistency import (
    SicdConsistency,
)
from ._sidd_consistency import (
    SiddConsistency,
)

__all__ = [
    "CphdConsistency",
    "CrsdConsistency",
    "SicdConsistency",
    "SiddConsistency",
]
