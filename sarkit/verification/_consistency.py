"""
Common functionality for verifying files for internal consistency.
"""

import argparse
import ast
import contextlib
import linecache
import os
import re
import sys
import textwrap
from collections.abc import Iterable
from typing import Optional

import numpy as np


class _ExceptionOnUse:
    """Class to help handle optional dependencies"""

    def __init__(self, exception):
        self._error_class = type(exception)
        self._error_message = exception.msg

    def __getattr__(self, name):
        raise self._error_class(self._error_message)


def _exception_stack():
    """Helper function to parse call stack of an exception

    Returns
    -------
    list of dict
        {'filename': str, 'lineno': int, 'line': str, 'context': dict} for each traceback in the current exception
    """

    try:
        _, _, tb = sys.exc_info()

        stack = []
        tback = tb
        while tback is not None:
            frame = tback.tb_frame
            filename = frame.f_code.co_filename
            linecache.checkcache(filename)
            inst = tback.tb_lasti // 2
            line_start, line_end = list(frame.f_code.co_positions())[inst][:2]
            lines = []
            for lineno in range(line_start, line_end + 1):
                lines.append(
                    linecache.getline(filename, lineno, frame.f_globals).strip()
                )
            line = " ".join(lines)

            ns = dict(frame.f_builtins)
            ns.update(frame.f_globals)
            ns.update(frame.f_locals)
            stack.append(
                {
                    "filename": filename,
                    "lineno": line_start,
                    "line": line.strip(),
                    "context": ns,
                }
            )
            tback = tback.tb_next

    finally:
        tb = None

    return stack


def _line_details(line, ns):
    try:
        thing = ast.parse(line)
    except Exception:
        return ""
    context = ""
    added_to_context = set()
    for node in ast.walk(thing):
        if isinstance(node, ast.Constant):
            continue
        unparsed = ast.unparse(node)
        if unparsed in added_to_context:
            continue
        added_to_context.add(unparsed)
        try:
            evaled = eval(unparsed, ns)
            context += f"\n\n where:\n {unparsed} = {evaled!r}"
        except Exception:
            pass
    return context


class ConsistencyChecker(object):
    """Base class for implementing consistency checkers.

    This class can be used to perform and log comparisons. Each comparison
    can be logged as either an ``'Error'`` or a ``'Warning'``.
    """

    def __init__(self):
        self._all_check_results = {}
        self._active_check = None

        names = [name for name in dir(self) if name.startswith("check_")]
        attrs = [getattr(self, name) for name in sorted(names)]
        self.funcs = [attr for attr in attrs if hasattr(attr, "__call__")]

    def check(
        self,
        func_name: Optional[str | Iterable[str]] = None,
        *,
        allow_prefix: bool = False,
        ignore_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        """Run checks.

        Parameters
        ----------
        func_name : str or list-like of str, optional
            List of check functions to run.  If omitted, then all check functions
            will be run.
        allow_prefix : bool
            If ``True``, runs tests with names starting with any ``func_name``
            If ``False``, runs tests with names equal to any ``func_name``
        ignore_patterns : list-like of str, optional
            Skips tests if zero or more characters at the beginning of their name match the regular expression patterns
        """
        # run specified test(s) or all of them
        if func_name is None:
            funcs = self.funcs
        else:
            if isinstance(func_name, str):
                func_name = [func_name]

            def matches_prefix(requested, actual):
                return actual.startswith(requested)

            def matches_exact(requested, actual):
                return requested == actual

            qualifier = matches_prefix if allow_prefix else matches_exact
            funcs = []
            not_found = []
            for requested_func in set(func_name):
                matches = [
                    func
                    for func in self.funcs
                    if qualifier(requested_func, func.__name__)
                ]
                funcs.extend(matches)
                if not matches:
                    not_found.append(requested_func)

            if not_found:
                raise ValueError(f"Functions not found: {not_found}")

        for pattern in ignore_patterns or []:
            funcs = [func for func in funcs if not re.match(pattern, func.__name__)]

        for func in funcs:
            self._run_check(func)

    def _run_check(self, func):
        """Runs a single 'check_' method and store the results.

        Parameters
        ----------
        func : Callable
            Run the supplied function
        """

        self._active_check = {"doc": func.__doc__, "details": [], "passed": True}

        # func() will populate self._active_check
        try:
            func()
        except Exception as e:
            stack = _exception_stack()
            message = []
            for indent, frame in enumerate(stack[1:]):
                message.append(
                    " " * indent * 4
                    + "line#{lineno}: {line}".format(
                        lineno=frame["lineno"], line=frame["line"]
                    )
                )
            message.append(str(e))
            self._add_item_to_current(
                "Error", False, "\n".join(message), details="Exception Raised"
            )

        self._all_check_results[func.__name__] = self._active_check
        self._active_check = None

    def _add_item_to_current(self, severity, passed, message, details=""):
        """Records the result of a test.

        Parameters
        ----------
        severity : str
            Severity level of the results e.g. 'Error', 'Warning'
        passed : bool
            The result of the test
        message : str
            Text message describing the test
        details : str
            Additional message details
        """

        item = {
            "severity": severity,
            "passed": passed,
            "message": message,
            "details": str(details),
        }

        self._active_check["details"].append(item)
        self._active_check["passed"] &= passed

    def _format_assertion(self, e, depth=1):
        """Format an assertion to human-readable text.

        Parameters
        ----------
        e : Exception
            The exception to be formatted
        depth : int
            Which level of the exception stack to format

        Returns
        -------
        formatted : str
            Formatted stack level containing line number and line text
        """

        stack = _exception_stack()
        frame = stack[depth]
        return (
            "line#{lineno}: {line}".format(lineno=frame["lineno"], line=frame["line"])
            + _line_details(frame["line"], frame["context"])
            + "\n"
            + "\n".join(str(x) for x in e.args)
        )

    @contextlib.contextmanager
    def need(self, details=None):
        """Context manager for scoping 'Error' level checks

        Parameters
        ----------
        details : str, optional
            Text describing the scope of checks
        """

        with self._crave("Error", details=details):
            yield

    @contextlib.contextmanager
    def want(self, details=None):
        """Context manager for scoping 'Warning' level checks

        Parameters
        ----------
        details : str, optional
            Text describing the scope of checks
        """

        with self._crave("Warning", details=details):
            yield

    @contextlib.contextmanager
    def _crave(self, level, details=None, depth=2):
        """Context manager for scoping checks

        Parameters
        ----------
        level : str
            Severity level of the checks.  e.g. 'Error' or 'Warning'
        details : str, optional
            Text describing the scope of checks
        depth : int
            Depth in the exception stack to look for check information
        """

        try:
            yield
            if self._active_check is not None:
                self._add_item_to_current(level, True, "", details=details)
        except AssertionError as e:
            if self._active_check is None:
                raise
            if not details:
                stack = _exception_stack()
                details = stack[depth]["line"]
            self._add_item_to_current(
                level, False, self._format_assertion(e, depth=depth), details=details
            )

    @contextlib.contextmanager
    def precondition(self, details=None):
        """Context manager for scoping conditional ('No-Op' level) checks

        Parameters
        ----------
        details : str, optional
            Text describing the scope of checks
        """

        try:
            yield
        except AssertionError as e:
            if self._active_check is None:
                return
            if not details:
                stack = _exception_stack()
                details = stack[1]["line"]
            self._add_item_to_current(
                "No-Op", True, self._format_assertion(e), details=details
            )

    def all(self):
        """Returns all results.

        Returns
        -------
        dict
            Unfiltered dictionary of all (Passed, Failed, Skipped) results
        """

        return self._all_check_results

    def failures(self, omit_passed_sub=False):
        """Returns failure results.

        Parameters
        ----------
        omit_passed_sub : bool
            If True, passed sub-checks will be omitted.

        Returns
        -------
        dict
            Dictionary containing only results of failed checks
        """

        retval = {}
        for k, v in self._all_check_results.items():
            if not v["passed"]:
                retval[k] = dict(v)
                if omit_passed_sub:
                    retval[k]["details"] = [d for d in v["details"] if not d["passed"]]
        return retval

    def passes(self):
        """Returns passed checks that are not wholly No-Op.

        Returns
        -------
        dict
            Dictionary containing checks that are not wholly No-Op
        """
        return {
            k: v
            for k, v in self.all().items()
            if v["passed"] and any(d["severity"] != "No-Op" for d in v["details"])
        }

    def skips(self, include_partial=False):
        """Returns passed checks that are No-Op.

        Parameters
        ----------
        include_partial : bool, optional
            Include checks that are partially No-Op? False by default.

        Returns
        -------
        dict
            Dictionary containing checks with No-Ops
        """
        func = any if include_partial else all
        return {
            k: v
            for k, v in self.all().items()
            if v["passed"] and func(d["severity"] == "No-Op" for d in v["details"])
        }

    def print_result(
        self,
        *,
        include_passed_asserts: bool = True,
        color: None | bool = None,
        include_passed_checks: bool = False,
        width: int = 120,
        skip_detail: bool = False,
        fail_detail: bool = False,
        pass_detail: bool = False,
    ) -> None:
        """Print results to stdout.

        Parameters
        ----------
        include_passed_asserts : bool
            Print asserts which passed
        color : bool, optional
            Colorize the output. If ``None``, checks for presence of ``NO_COLOR`` environment variable and when absent
            colorizes if stdout is tty(-like).
        include_passed_checks : bool
            Print checks which passed
        width : int
            Output up to ``width`` columns
        skip_detail : bool
            Include details of skips
        fail_detail : bool
            Include details of failures
        pass_detail : bool
            Include details of passes
        """

        if color is None:
            color = (
                sys.stdout.isatty() and "NO_COLOR" not in os.environ
            )  # https://no-color.org
        to_print = {}
        for k, v in self._all_check_results.items():
            if include_passed_checks or not v["passed"]:
                to_print[k] = dict(v)
                to_print[k]["details"] = [
                    d for d in v["details"] if include_passed_asserts or not d["passed"]
                ]

        if color:
            coloration = {
                ("Error", True): ["[Pass]", "green", "bold"],
                ("Error", False): ["[Error]", "red", "bold"],
                ("Warning", True): ["[Pass]", "cyan"],
                ("Warning", False): ["[Warning]", "yellow"],
                ("No-Op", True): ["[Skip]", "blue"],
            }
        else:
            coloration = {
                ("Error", True): ["[Need]"],
                ("Error", False): ["[Error]"],
                ("Warning", True): ["[Want]"],
                ("Warning", False): ["[Warning]"],
                ("No-Op", True): ["[Skip]"],
            }

        indent = 4
        for case, details in to_print.items():
            print(f"{case}: {str(details['doc']).strip()}")
            if details["details"]:
                for sub in details["details"]:
                    lead = in_color(*coloration[sub["severity"], sub["passed"]])
                    need_want = {"Error": "Need", "Warning": "Want", "No-Op": "Unless"}[
                        sub["severity"]
                    ]
                    print(
                        "{indent}{lead} {need_want}: {details}".format(
                            indent=" " * indent,
                            lead=lead,
                            need_want=need_want,
                            details=sub["details"],
                        )
                    )
                    if (
                        skip_detail
                        and sub["severity"] == "No-Op"
                        or (fail_detail and not sub["passed"])
                        or (pass_detail and sub["passed"])
                    ):
                        for line in sub["message"].splitlines():
                            message = "\n".join(
                                textwrap.wrap(
                                    line,
                                    width=width,
                                    subsequent_indent=" " * (indent + 8),
                                    initial_indent=" " * (indent + 4),
                                )
                            )
                            print(message)
            else:
                print("{}---: No test performed".format(" " * indent))

    @staticmethod
    def add_cli_args(parser):
        """Add CLI args used by `run_cli` to an argparser"""
        parser.add_argument(
            "--ignore",
            action="extend",
            nargs="+",
            metavar="PATTERN",
            help=(
                "Skip any check matching PATTERN at the beginning of its name. "
                "Can be specified more than once."
            ),
        )
        parser.add_argument(
            "-v",
            "--verbose",
            default=0,
            action="count",
            help="Increase verbosity (can be specified more than once >4 doesn't help)",
        )
        parser.add_argument(
            "--array-limit",
            type=int,
            default=10,
            help="Number of array elements above which arrays are abbreviated",
        )
        parser.add_argument(
            "--color",
            action=argparse.BooleanOptionalAction,
            help="colorize output",
        )

    def run_cli(self, config):
        """Run checks and print results using args from `add_cli_args`"""
        with np.printoptions(threshold=config.array_limit):
            self.check(ignore_patterns=config.ignore)
            self.print_result(
                fail_detail=config.verbose >= 1,
                include_passed_asserts=config.verbose >= 2,
                include_passed_checks=config.verbose >= 3,
                skip_detail=config.verbose >= 4,
                color=config.color,
            )
            return bool(self.failures())


class Approx:
    """Wrapper for performing approximate value comparisons.

    Parameters
    ----------
    value : float
        The Value to be compared.
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance

    See Also
    --------
    pytest.approx
    """

    # Tell numpy to use our comparison operators
    __array_ufunc__ = None
    __array_priority__ = 100

    def __init__(self, value, atol=1e-10, rtol=1e-6):
        self.value = value
        self.atol = atol
        self.rtol = rtol

    def __lt__(self, rhs):
        return self.__le__(rhs)

    def __le__(self, rhs):
        return np.all(np.logical_or(np.less(self.value, rhs), self._isclose(rhs)))

    def __eq__(self, rhs):
        return np.all(self._isclose(rhs))

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def __ge__(self, rhs):
        return np.all(np.logical_or(np.greater(self.value, rhs), self._isclose(rhs)))

    def __gt__(self, rhs):
        return self.__ge__(rhs)

    def __repr__(self):
        tol = self.atol + np.abs(np.asarray(self.value)) * self.rtol
        return f"{self.value} ± {tol}"

    def _isclose(self, rhs):
        return np.isclose(rhs, self.value, rtol=self.rtol, atol=self.atol)


def in_color(string: str, *color: str) -> str:
    """Wrap a string with ANSI color control characters.

    Parameters
    ----------
    string : str
        The string to colorize.
    *color : str
        color identifiers to use.  See `start_color`.

    Returns
    -------
    str
        ANSI colorized version of `string`
    """

    if color:
        start = "".join(start_color(c) for c in color)
        return "{}{}{}".format(start, string, END_COLOR)
    else:
        return string


END_COLOR = "\x1b[0m"


def start_color(color: str) -> str:
    """Get an ANSI color control character.

    Parameters
    ----------
    color : {'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white', 'bold', 'light', 'invert'}
        Desired color

    Returns
    -------
    str
        ANSI color control for desired color
    """

    color_table = dict(
        black=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        purple=35,
        cyan=36,
        white=37,
        bold=1,
        light=2,
        invert=7,
    )
    return "\x1b[%sm" % color_table[color]
