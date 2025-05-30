import argparse
import contextlib
import filecmp
import pathlib
import sys
import tempfile

import lxml.etree
import xmlschema

import sarkit._xmlhelp2 as skxh2
import sarkit.sidd as sksidd


def generate_xsdtypes(xs: xmlschema.XMLSchema):
    assert len(xs.root_elements) == 1
    xsdtypes = {"/": make_typedef(xs.root_elements[0].type)}

    def process_elem(elem):
        typname = get_typename(elem)
        if typname not in xsdtypes:
            newitem = make_typedef(elem.type)
            xsdtypes[typname] = newitem

            for child in elem.iterchildren():
                process_elem(child)

            if newitem.text_typename and newitem.text_typename not in xsdtypes:
                qn = lxml.etree.QName(newitem.text_typename)
                root_typeobj = xs.get_schema(qn.namespace).types[qn.localname]
                xsdtypes[newitem.text_typename] = make_typedef(root_typeobj)

    process_elem(xs.root_elements[0])

    return xsdtypes


def get_typename(elemobj: xmlschema.XsdElement):
    return elemobj.type.name or (
        "<UNNAMED>-"
        + "/".join(
            x.name
            for x in reversed([elemobj] + list(elemobj.iter_ancestors()))
            if x.name
        )
    )


def make_childdef(elemobj: xmlschema.XsdElement):
    return skxh2.ChildDef(
        tag=elemobj.name,
        typename=get_typename(elemobj),
        repeat=elemobj.max_occurs is None or elemobj.max_occurs > 1,
    )


def make_typedef(typeobj: xmlschema.XsdType):
    kwargs = {}
    if typeobj.has_simple_content() or typeobj.has_mixed_content():
        if typeobj.base_type is not None:
            # Best effort; if None, something downstream will have to deal with it (if desired)
            # this includes unions, lists, etc.
            kwargs["text_typename"] = typeobj.base_type.name
    if typeobj.is_element_only() or typeobj.has_mixed_content():
        kwargs["children"] = [make_childdef(c) for c in typeobj.content.iter_elements()]
    return skxh2.XsdTypeDef(
        attributes=list(getattr(typeobj, "attributes", [])), **kwargs
    )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't write the files, just return the status. Return code 0 means nothing would change.",
    )
    config = parser.parse_args(args)

    schema_files = [x["schema"] for x in sksidd.VERSION_INFO.values()]

    files_that_differ = set()
    for schema in schema_files:
        xs = xmlschema.XMLSchema(schema)
        xsdtypes = generate_xsdtypes(xs)
        # put xsdtypes folder as sibling to schemas folder
        schemas_dir = next(p for p in schema.parents if p.stem == "schemas")
        storage_dir = schemas_dir.parent / "xsdtypes"

        with (
            tempfile.TemporaryDirectory()
            if config.check
            else contextlib.nullcontext(storage_dir)
        ) as outdir:
            outdir = pathlib.Path(outdir)
            outdir.mkdir(exist_ok=True)
            output_file = outdir / f"{schema.stem}.json"
            output_file.write_text(skxh2.dumps_xsdtypes(xsdtypes))

            if config.check and not filecmp.cmp(
                storage_dir / output_file.name, output_file
            ):
                files_that_differ.add(output_file.name)

    return len(files_that_differ)


if __name__ == "__main__":
    sys.exit(main())
