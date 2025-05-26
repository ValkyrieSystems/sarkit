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

   $ cphd-consistency /path/to/file
   $ crsd-consistency /path/to/file
   $ sicd-consistency /path/to/file
   $ sidd-consistency /path/to/file

The command line flags for each are given below:

.. _cphd-consistency-cli:

.. autoprogram:: sarkit.verification._cphd_consistency:_parser()
   :prog: cphd-consistency

.. _crsd-consistency-cli:

.. autoprogram:: sarkit.verification._crsd_consistency:_parser()
   :prog: crsd-consistency

.. _sicd-consistency-cli:

.. autoprogram:: sarkit.verification._sicd_consistency:_parser()
   :prog: sicd-consistency

.. _sidd-consistency-cli:

.. autoprogram:: sarkit.verification._sidd_consistency:_parser()
   :prog: sidd-consistency
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
