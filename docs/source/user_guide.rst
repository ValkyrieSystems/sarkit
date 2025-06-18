.. _user_guide:

=================
SARkit User Guide
=================

:Release: |version|
:Date: |today|

SARkit contains readers and writers for SAR standards files and functions for operating on them.
This is an overview of basic SARkit functionality. For details, see the :doc:`reference/index`.

.. _installation:

Installation
============
Basic SARkit functionality relies on a small set of dependencies.
Some features require additional dependencies which can be installed using packaging extras:

.. code-block:: shell-session

   $ python -m pip install sarkit  # Install core dependencies
   $ python -m pip install sarkit[processing]  # Install processing dependencies
   $ python -m pip install sarkit[verification]  # Install verification dependencies
   $ python -m pip install sarkit[all]  # Install all dependencies


Reading and writing files
=========================
SARkit provides reader/writer classes that are intended to be used as context managers and metadata classes that are
used to describe settable metadata.

.. list-table::

   * - Format
     - Reader
     - Metadata
     - Writer
   * - CRSD
     - :py:class:`~sarkit.crsd.Reader`
     - :py:class:`~sarkit.crsd.Metadata`
     - :py:class:`~sarkit.crsd.Writer`
   * - CPHD
     - :py:class:`~sarkit.cphd.Reader`
     - :py:class:`~sarkit.cphd.Metadata`
     - :py:class:`~sarkit.cphd.Writer`
   * - SICD
     - :py:class:`~sarkit.sicd.NitfReader`
     - :py:class:`~sarkit.sicd.NitfMetadata`
     - :py:class:`~sarkit.sicd.NitfWriter`
   * - SIDD
     - :py:class:`~sarkit.sidd.NitfReader`
     - :py:class:`~sarkit.sidd.NitfMetadata`
     - :py:class:`~sarkit.sidd.NitfWriter`


Reading
-------

Readers are instantiated with a `file object` and file contents are accessed via the ``metadata`` attribute and
format-specific methods.
In general, only the container information is accessed upon instantiation; further file access is deferred until
data access methods are called.
This pattern makes it faster to read components out of large files and is especially valuable for metadata access which
is often a small fraction of the size of a SAR data file.

.. testsetup::

   import lxml.etree
   import numpy as np

   import sarkit.sicd as sksicd

   example_sicd = tmppath / "example.sicd"
   sec = {"security": {"clas": "U"}}
   parser = lxml.etree.XMLParser(remove_blank_text=True)
   example_sicd_xmltree = lxml.etree.parse("data/example-sicd-1.4.0.xml", parser)
   sicd_meta = sksicd.NitfMetadata(
       xmltree=example_sicd_xmltree,
       file_header_part={"ostaid": "nowhere", "ftitle": "SARkit example SICD FTITLE"} | sec,
       im_subheader_part={"isorce": "this sensor"} | sec,
       de_subheader_part=sec,
   )
   with open(example_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta):
       pass  # don't currently care about the pixels

.. doctest::

   >>> with example_sicd.open("rb") as f, sksicd.NitfReader(f) as reader:
   ...     pixels = reader.read_image()
   ...     pixels.shape
   (5727, 2362)

   # Metadata, but not methods, can be safely accessed outside of the
   # context manager's context

   # Access specific NITF fields that are called out in the SAR standards
   >>> reader.metadata.file_header_part.ftitle
   'SARkit example SICD FTITLE'

   # XML metadata is returned as lxml.etree.ElementTree objects
   >>> (reader.metadata.xmltree.findtext(".//{*}FullImage/{*}NumRows"),
   ...  reader.metadata.xmltree.findtext(".//{*}FullImage/{*}NumCols"))
   ('5727', '2362')


Metadata
--------

``Metadata`` objects contain all of the standard-specific settable metadata.
This includes XML instance(s) and container metadata (PDD-settable NITF fields, CPHD header fields, etc.).

Metadata objects can be built from their components:

.. doctest::

   >>> new_metadata = sksicd.NitfMetadata(
   ...     xmltree=example_sicd_xmltree,
   ...     file_header_part={"ostaid": "my location", "security": {"clas": "U"}},
   ...     im_subheader_part={"isorce": "my sensor", "security": {"clas": "U"}},
   ...     de_subheader_part={"security": {"clas": "U"}},
   ... )

Metadata objects are also available from readers:

.. doctest::

   >>> read_metadata = reader.metadata


Writing
-------

Writers are instantiated with a `file object` and a ``Metadata`` object.
SARkit relies on upfront metadata because for many of the SAR standards it is more efficient to know what a file will
contain before writing.
Similar to reading, instantiating a writer sets up the file while data is written using format-specific methods.

.. doctest::

   >>> written_sicd = tmppath / "written.sicd"
   >>> with written_sicd.open("wb") as f, sksicd.NitfWriter(f, read_metadata) as writer:
   ...     writer.write_image(pixels)

   >>> with written_sicd.open("rb") as f:
   ...     f.read(9).decode()
   'NITF02.10'

SARkit sanity checks some aspects on write but it is up to the user to ensure consistency of the metadata and data:

.. doctest::

   >>> bad_sicd = tmppath / "bad.sicd"
   >>> with bad_sicd.open("wb") as f, sksicd.NitfWriter(f, read_metadata) as writer:
   ...     writer.write_image(pixels.view(np.uint8))
   Traceback (most recent call last):
   ValueError: Array dtype (uint8) does not match expected dtype (complex64) for PixelType=RE32F_IM32F

SARkit provides :ref:`consistency checkers <consistency_checking>` that can be used to help create self-consistent SAR
data.


Operating on XML Metadata
=========================
The parsed XML element tree is a key component in SARkit as XML is the primary metadata container for many SAR
standards.

For simple operations, `xml.etree.ElementTree` and/or `lxml` are often sufficient:

.. doctest::

   >>> example_sicd_xmltree.findtext(".//{*}ModeType")
   'SPOTLIGHT'

For complicated metadata, SARkit provides XML helper classes that can be used to transcode between XML and more
convenient Python objects.

.. list-table::

   * - Format
     - XML Helper
   * - CRSD
     - :py:class:`sarkit.crsd.XmlHelper`
   * - CPHD
     - :py:class:`sarkit.cphd.XmlHelper`
   * - SICD
     - :py:class:`sarkit.sicd.XmlHelper`
   * - SIDD
     - :py:class:`sarkit.sidd.XmlHelper`

XML Helpers
-----------

XmlHelpers are instantiated with an `lxml.etree.ElementTree` which can then be manipulated using set and load methods.

.. doctest::

   >>> import sarkit.sicd as sksicd
   >>> xmlhelp = sksicd.XmlHelper(example_sicd_xmltree)
   >>> xmlhelp.load(".//{*}ModeType")
   'SPOTLIGHT'

:py:class:`~sarkit.sicd.XmlHelper.load_elem` and :py:class:`~sarkit.sicd.XmlHelper.set_elem`
can be used when you already have an element object:

.. doctest::

   >>> tcoa_poly_elem = example_sicd_xmltree.find(".//{*}TimeCOAPoly")
   >>> xmlhelp.load_elem(tcoa_poly_elem)
   array([[1.2206226]])

   >>> xmlhelp.set_elem(tcoa_poly_elem, [[1.1, -2.2], [-3.3, 4.4]])
   >>> print(lxml.etree.tostring(tcoa_poly_elem, pretty_print=True, encoding="unicode").strip())
   <TimeCOAPoly xmlns="urn:SICD:1.4.0" order1="1" order2="1">
     <Coef exponent1="0" exponent2="0">1.1</Coef>
     <Coef exponent1="0" exponent2="1">-2.2</Coef>
     <Coef exponent1="1" exponent2="0">-3.3</Coef>
     <Coef exponent1="1" exponent2="1">4.4</Coef>
   </TimeCOAPoly>

:py:class:`~sarkit.sicd.XmlHelper.load` / :py:class:`~sarkit.sicd.XmlHelper.set` are
shortcuts for ``find`` + :py:class:`~sarkit.sicd.XmlHelper.load_elem` /
:py:class:`~sarkit.sicd.XmlHelper.set_elem`:

.. doctest::

   # find + set_elem/load_elem
   >>> elem = example_sicd_xmltree.find("{*}ImageData/{*}SCPPixel")
   >>> xmlhelp.set_elem(elem, [123, 456])
   >>> xmlhelp.load_elem(elem)
   array([123, 456])

   # equivalent methods using set/load
   >>> xmlhelp.set("{*}ImageData/{*}SCPPixel", [321, 654])
   >>> xmlhelp.load("{*}ImageData/{*}SCPPixel")
   array([321, 654])

.. note:: Similar to writers, XMLHelpers only prevent basic errors. Users are responsible for ensuring metadata is
   accurate and compliant with the standard/schema.


What is transcodable?
---------------------

Every leaf in the supported SAR standards' XML trees has a transcoder, but parent nodes generally only have them for
standard-defined complex types (e.g. XYZ, LL, LLH, POLY, 2D_POLY, etc.).
Select parent nodes also have them when a straightforward mapping is apparent (e.g. polygons).

.. doctest::

   # this leaf has a transcoder
   >>> xmlhelp.load("{*}CollectionInfo/{*}CollectorName")
   'SyntheticCollector'

   # this parent node does not have a transcoder
   >>> xmlhelp.load("{*}CollectionInfo")
   Traceback (most recent call last):
   LookupError: CollectionInfo is not transcodable


.. _consistency_checking:

Consistency Checking
====================

.. warning:: Consistency checkers require the ``verification`` :ref:`extra <installation>`.

SARkit provides checkers that can be used to identify inconsistencies in SAR standards files.

.. list-table::

   * - Format
     - Consistency class
     - Command
   * - CRSD
     - :py:class:`sarkit.verification.CrsdConsistency`
     - :ref:`crsd-consistency-cli`
   * - CPHD
     - :py:class:`sarkit.verification.CphdConsistency`
     - :ref:`cphd-consistency-cli`
   * - SICD
     - :py:class:`sarkit.verification.SicdConsistency`
     - :ref:`sicd-consistency-cli`
   * - SIDD
     - :py:class:`sarkit.verification.SiddConsistency`
     - :ref:`sidd-consistency-cli`

Each consistency checker provides a command line interface for checking SAR data/metadata files.
When there are no inconsistencies, no output is produced.

.. code-block:: shell-session

   $ sicd-consistency good.sicd
   $

The same command can be used to run a subset of the checks against the XML.

.. code-block:: shell-session

   $ sicd-consistency good.sicd.xml
   $

When a file is inconsistent, failed checks are printed.

.. code-block:: shell-session

   $ sicd-consistency bad.sicd
   check_image_formation_timeline: Checks that the slow time span for data processed to form
   the image is within collect.
      [Error] Need: 0 <= TStartProc < TEndProc <= CollectDuration

For further details about consistency checker results, increase the output verbosity.
The ``-v`` flag is additive and can be used up to 4 times.

.. code-block::

   -v       # display details in failed checks
   -vv      # display passed asserts in failed checks
   -vvv     # display passed checks
   -vvvv    # display details in skipped checks

For example:

.. code-block:: shell-session

   $ sicd-consistency good.sicd -vvv
   check_against_schema: Checks against schema.
      [Pass] Need: XML passes schema
      [Pass] Need: Schema available for checking xml whose root tag = {urn:SICD:1.2.1}SICD
   ...
