"""Wave 1 acceptance: the package imports and exposes a version."""


def test_package_exposes_version() -> None:
    import pawbench

    assert isinstance(pawbench.__version__, str)
    assert pawbench.__version__.count(".") == 2
