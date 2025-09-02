import numpy.testing as npt


def elem_cmp(a, b, xsdhelper):
    if a.tag != b.tag:
        return False
    try:
        npt.assert_equal(a.attrib, b.attrib)
    except AssertionError:
        return False

    t = xsdhelper.get_elem_transcoder(a)
    if len(a) == len(b) == 0:
        try:
            npt.assert_equal(t.parse_elem(a), t.parse_elem(b))
        except (AssertionError, AttributeError):
            # sometimes text is None or has pretty-printing
            a_text = (a.text or "").strip()
            b_text = (b.text or "").strip()
            return a_text == b_text
        return True

    b_children = list(b)
    for a_child in a:
        for b_child in b_children:
            if elem_cmp(a_child, b_child, xsdhelper):
                b_children.remove(b_child)
                break
        else:
            return False
    return all(float(x.text) == 0 for x in b_children)
