import importlib


def test_package_import():
    package = importlib.import_module("audio_classification")

    assert package.__file__ is not None
