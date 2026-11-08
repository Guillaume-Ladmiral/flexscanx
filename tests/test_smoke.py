import flexscanx as p


def test_smoke_import():
    v = p.core_version()
    assert isinstance(v, str)
