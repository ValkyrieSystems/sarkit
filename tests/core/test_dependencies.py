import importlib
import pkgutil
import unittest

import sarkit

NEEDS_EXTRAS = ["sarkit._processing"]


class TestImports(unittest.TestCase):
    def test_can_import(self):
        def handle_import_error(failed_import_name):
            with self.subTest(f"Failed to import {failed_import_name}"):
                self.assertIn(failed_import_name, NEEDS_EXTRAS)

        for modinfo in pkgutil.walk_packages(
            sarkit.__path__, sarkit.__name__ + ".", onerror=handle_import_error
        ):
            if modinfo.name in NEEDS_EXTRAS:
                continue
            with self.subTest(name=modinfo.name):
                module = importlib.import_module(modinfo.name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
