import itertools
import re


def assert_failures(con, pattern):
    pattern = re.compile(pattern)
    failure_details = itertools.chain(
        *[x["details"] for x in con.failures(omit_passed_sub=True).values()]
    )
    failure_messages = [x["details"] for x in failure_details]

    # this construction can help improve the error message for determining why pattern is not present
    assert any(list(map(pattern.search, failure_messages)))
