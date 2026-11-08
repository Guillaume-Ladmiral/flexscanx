from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIGNETTE = ROOT / "vignettes" / "workflows" / "rflexscan_manual_examples.qmd"
DATA = ROOT / "vignettes" / "North_Carolina_SIDS.gpkg"
WEIGHTS = ROOT / "vignettes" / "ncCR85.gal"


def test_rflexscan_manual_vignette_has_self_contained_original_data() -> None:
    assert DATA.is_file()
    assert WEIGHTS.is_file()
    assert DATA.stat().st_size > 100_000
    assert WEIGHTS.read_text(encoding="utf-8").startswith("0 100 sids rn\n")
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == (
        "67ab56c847e1524385adf3b4e67648054035d137e5fe956f96e16941bbc5a00f"
    )
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == (
        "d042127f8c5bd1a1e2b3e5f29c898cbec0d37ee1f3b7b93e48f4d6724d413ddb"
    )


def test_rflexscan_manual_vignette_covers_every_worked_example_group() -> None:
    source = VIGNETTE.read_text(encoding="utf-8")

    required_sections = (
        "# Coverage of the original manual",
        "# The scan example",
        "## Print the complete result",
        "## Print the most likely cluster",
        "## Print the summary",
        "# Network plots",
        "## All detected clusters as a network",
        "## Network plot for ranks 1–3",
        "## Network plot for p-values at or below 0.05",
        "# Choropleth maps",
        "## All detected clusters on the county map",
        "## County map for ranks 1–3",
        "## County map for p-values at or below 0.05",
    )
    for section in required_sections:
        assert section in source

    required_code = (
        'gdf["BIR74"] * gdf["SID74"].sum() / gdf["BIR74"].sum()',
        "flx.scan(",
        'scanmethod="FLEXIBLE"',
        'stattype="ORIGINAL"',
        "clustersize=10",
        'secondary="ALL"',
        "flx.summarize_result(result)",
        "display(primary_cluster)",
        "plot_cluster_network(result, ranks=None, pvalue_max=None)",
        "plot_cluster_network(result, ranks=(1, 2, 3), pvalue_max=None)",
        "plot_cluster_network(result, ranks=None, pvalue_max=0.05)",
        "plot_cluster_choropleth(result, ranks=None, pvalue_max=None)",
        "plot_cluster_choropleth(result, ranks=(1, 2, 3), pvalue_max=None)",
        "plot_cluster_choropleth(result, ranks=None, pvalue_max=0.05)",
    )
    for snippet in required_code:
        assert snippet in source

    assert "```{python}" in source
    assert "execute:\n  enabled: true" in source


def test_rflexscan_manual_vignette_is_registered_as_a_user_workflow() -> None:
    index = (ROOT / "vignettes" / "README.md").read_text(encoding="utf-8")
    documentation_index = (ROOT / "DOCUMENTATION_INDEX.md").read_text(encoding="utf-8")

    assert "workflows/rflexscan_manual_examples.qmd" in index
    assert "rflexscan_manual_examples.qmd" in documentation_index
