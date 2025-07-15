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

For complicated metadata, SARkit provides helper classes that make manipulating and using XML more convenient.

.. list-table::

   * - `XML Helpers`_
     - Transcode between XML elements and convenient Python objects
   * - `XSD Helpers`_
     - Retrieve transcoders and type information for elements of a given XML schema
   * - `Element Wrappers`_
     - Access metadata via a dictionary-like interface

Helper classes are provided for each SAR standard:

.. list-table::

   * - Format
     - XML Helper
     - XSD Helper
     - Element Wrapper
   * - CRSD
     - :py:class:`~sarkit.crsd.XmlHelper`
     - :py:class:`~sarkit.crsd.XsdHelper`
     - :py:class:`~sarkit.crsd.ElementWrapper`
   * - CPHD
     - :py:class:`~sarkit.cphd.XmlHelper`
     - :py:class:`~sarkit.cphd.XsdHelper`
     - :py:class:`~sarkit.cphd.ElementWrapper`
   * - SICD
     - :py:class:`~sarkit.sicd.XmlHelper`
     - :py:class:`~sarkit.sicd.XsdHelper`
     - :py:class:`~sarkit.sicd.ElementWrapper`
   * - SIDD
     - :py:class:`~sarkit.sidd.XmlHelper`
     - :py:class:`~sarkit.sidd.XsdHelper`
     - :py:class:`~sarkit.sidd.ElementWrapper`

XML Helpers
-----------

XmlHelpers transcode between XML and more convenient Python objects.
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
   LookupError: {urn:SICD:1.4.0}CollectionInfo is not transcodable


XSD Helpers
-----------

XsdHelpers retrieve transcoders and type information for elements of a given XML schema.
XsdHelpers are instantiated by their target namespace.

.. doctest::

   >>> xsdhelp = sksicd.XsdHelper("urn:SICD:1.4.0")

   # retrieve transcoder by type name
   >>> transcoder = xsdhelp.get_transcoder("{urn:SICD:1.4.0}PolygonType")

   # retrieve the type name and definition for an element
   >>> typename, typedef = xsdhelp.get_elem_typeinfo(example_sicd_xmltree.find("{*}GeoData/{*}ValidData"))
   >>> print(typename)
   {urn:SICD:1.4.0}PolygonType
   >>> import pprint
   >>> pprint.pprint(typedef)
   XsdTypeDef(attributes=['size'],
              children=[ChildDef(tag='{urn:SICD:1.4.0}Vertex',
                                 typename='<UNNAMED>-{urn:SICD:1.4.0}PolygonType/{urn:SICD:1.4.0}Vertex',
                                 repeat=True)],
              text_typename=None)


Element Wrappers
----------------

ElementWrappers wrap an `lxml.etree.Element` to provide a dictionary-like interface.

Child elements of the wrapped element are keyed by local names.
Namespaces and element ordering are handled automatically.

When the child being accessed is not transcodable, a new ElementWrapper is returned.

.. doctest::

   >>> wrappedsicd = sksicd.ElementWrapper(example_sicd_xmltree.getroot())
   >>> wrappedsicd["ImageCreation"]
   ElementWrapper({'Application': 'Valkyrie Systems Sage | sar_common_kit 1.12.7.0', 'DateTime': datetime.datetime(2024, 5, 29, 14, 14, 28, 527158, tzinfo=datetime.timezone.utc)})

When the child being accessed is transcodable, the transcoded value is returned.

.. doctest::

   >>> wrappedsicd["Grid"]["Row"]["UVectECF"]
   array([-6.32466683e-01, -1.87853957e-06, -7.74587565e-01])

.. note:: Transcoded values are copies, not references. Some effort has been made to make them immutable.

Repeatable elements are treated as tuples.

.. doctest::

   >>> for p in wrappedsicd["ImageFormation"]["Processing"]:
   ...    print(p["Type"])
   inscription
   Valkyrie Systems Sage | sar_common_kit 1.12.7.0 @ 2024-05-29T14:12:54.542381Z
   polar_deterministic_phase

Accessing keys that are not schema-valid raises a `KeyError`:

.. doctest::

   >>> wrappedsicd["NotValid"]
   Traceback (most recent call last):
   KeyError: 'NotValid'

Accessing schema-valid keys that don't exist does not raise an exception.

.. doctest::

   >>> wrappedsicd["RMA"]
   ElementWrapper({})

Attributes are accessed using BadgerFish notation (e.g. @attr).

.. doctest::

   >>> wrappedsicd["RadarCollection"]["Area"]["Plane"]["RefPt"]["@name"]
   'Null Island'

Children can be set using ElementWrappers, `lxml.etree.Element` s, dictionaries, or - if transcodable - the
transcoded values.

.. doctest::

   # set item using an ElementWrapper
   >>> wrapped_tx = wrappedsicd["Antenna"]["Tx"]
   >>> wrappedsicd["Antenna"]["Rcv"] = wrapped_tx

   # set item using an lxml.etree.Element
   >>> manual_elem = lxml.etree.Element("{urn:SICD:1.4.0}FreqZero")
   >>> manual_elem.text = "24.0"
   >>> wrappedsicd["Antenna"]["Rcv"]["FreqZero"] = manual_elem

   # set item using a dict
   >>> wrappedsicd["Antenna"]["Rcv"]["EB"] = {"DCXPoly": [0.0], "DCYPoly": [1.0, 2.0]}

   # set item using a transcoded value
   >>> wrappedsicd["Antenna"]["Rcv"]["XAxisPoly"] = np.arange(12).reshape((-1, 3))

Non-existent ancestors are created as necessary.

.. doctest::

   >>> del wrappedsicd["CollectionInfo"]
   >>> wrappedsicd.elem.find("{*}CollectionInfo") is None
   True
   >>> wrappedsicd["CollectionInfo"]["RadarMode"]["ModeType"] = "SPOTLIGHT"
   >>> lxml.etree.dump(wrappedsicd["CollectionInfo"].elem)
   <CollectionInfo xmlns="urn:SICD:1.4.0">
     <RadarMode>
       <ModeType>SPOTLIGHT</ModeType>
     </RadarMode>
   </CollectionInfo>

Use :py:meth:`~sarkit.xmlhelp.ElementWrapper.add` to add repeatable children.

.. doctest::

   >>> len(wrappedsicd["CollectionInfo"]["CountryCode"])
   0
   >>> for cc in ("AB", "CD", "EF"):
   ...     _ = wrappedsicd["CollectionInfo"].add("CountryCode", cc)
   >>> wrappedsicd["CollectionInfo"]["CountryCode"]
   ('AB', 'CD', 'EF')

To serialize and deserialize ElementWrappers, use :py:meth:`~sarkit.xmlhelp.ElementWrapper.to_dict` and
:py:meth:`~sarkit.xmlhelp.ElementWrapper.from_dict`:

.. doctest::

   >>> d = wrappedsicd["CollectionInfo"].to_dict()
   >>> print(d)
   {'RadarMode': {'ModeType': 'SPOTLIGHT'}, 'CountryCode': ('AB', 'CD', 'EF')}

   >>> wrappedsicd["CollectionInfo"].from_dict(d | {"CollectorName": "coll", "IlluminatorName": "illum"})
   >>> lxml.etree.dump(wrappedsicd["CollectionInfo"].elem)
   <CollectionInfo xmlns="urn:SICD:1.4.0">
     <CollectorName>coll</CollectorName>
     <IlluminatorName>illum</IlluminatorName>
     <RadarMode>
       <ModeType>SPOTLIGHT</ModeType>
     </RadarMode>
     <CountryCode>AB</CountryCode>
     <CountryCode>CD</CountryCode>
     <CountryCode>EF</CountryCode>
   </CollectionInfo>

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
