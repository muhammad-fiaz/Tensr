"""Conan recipe module for the Tensr library.

This module defines a Conan recipe (`TensrConan`) that builds the project
using xmake and packages headers and libraries for consumers. It also
contains small fallbacks so the file can be opened in editors that don't
have Conan installed in the environment.
"""

from typing import TYPE_CHECKING

# Try to import Conan at runtime; editors or environments without Conan
# will use the TYPE_CHECKING branch for accurate typings and a minimal
# runtime fallback to avoid import errors from linters (E0401).
if TYPE_CHECKING:
    # For static type checkers and IDEs that have Conan available via
    # the project's dev dependencies, keep the real imports.
    from conan import ConanFile  # type: ignore
    from conan.tools.files import copy  # type: ignore
else:
    try:
        from conan import ConanFile  # type: ignore
        from conan.tools.files import copy  # type: ignore
    except (
        ImportError,
        ModuleNotFoundError,
    ):  # pragma: no cover - fallback for editor environments
        # Minimal placeholders so the file can be opened and linted where
        # Conan isn't installed. These will not be used when running under
        # Conan during packaging/build.
        class ConanFile:  # pragma: no cover - simple placeholder
            """Minimal placeholder for ConanFile when Conan isn't installed.

            This allows editors and linters to open the file without raising
            import errors. The real `ConanFile` from Conan will be used when
            running within a Conan environment.
            """

        def copy(*args, **kwargs):  # pragma: no cover - placeholder
            """Placeholder for `conan.tools.files.copy` when Conan isn't installed.

            This raises at runtime if invoked; it's present only so editors and
            linters can import this module when Conan isn't available.
            """
            raise RuntimeError("Conan is not installed; 'copy' is unavailable")


class TensrConan(ConanFile):
    """Conan recipe for the Tensr library.

    Builds the project using xmake and packages headers and static/shared
    libraries for consumers.
    """

    name = "tensr"
    version = "0.0.0"
    license = "Apache-2.0"
    author = "Muhammad Fiaz <contact@muhammadfiaz.com>"
    url = "https://github.com/muhammad-fiaz/tensr"
    description = "A powerful, superfast multidimensional tensor library for C/C++"
    topics = ("tensor", "machine-learning", "scientific-computing", "gpu")
    settings = "os", "compiler", "build_type", "arch"
    options = {"cuda": [True, False]}
    default_options = {"cuda": True}
    exports_sources = "src/*", "include/*", "xmake.lua"

    def build(self):
        """Build the Tensr library using xmake.

        Configures a release build and builds the `tensr` target.
        """
        self.run("xmake config -m release")
        self.run("xmake build tensr")

    def package(self):
        """Package headers and libraries into the Conan package folder.

        Copies public headers from the source tree and static or import
        libraries from the build folder into `self.package_folder`.
        """
        copy(
            self, "*.h", src=self.source_folder, dst=self.package_folder, keep_path=True
        )
        copy(
            self,
            "*.hpp",
            src=self.source_folder,
            dst=self.package_folder,
            keep_path=True,
        )
        copy(
            self, "*.a", src=self.build_folder, dst=self.package_folder, keep_path=False
        )
        copy(
            self,
            "*.lib",
            src=self.build_folder,
            dst=self.package_folder,
            keep_path=False,
        )

    def package_info(self):
        """Provide package info to consumers: library names and link flags.

        Consumers will link against the `tensr` library.
        """
        self.cpp_info.libs = ["tensr"]
