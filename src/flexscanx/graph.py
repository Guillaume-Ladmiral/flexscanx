from __future__ import annotations

import os
from typing import Iterable

import numpy as np

__all__ = [
    "csr_from_pysal",
    "csr_from_dense",
    "csr_knn_from_coords",
    "queen_adjacency",
    "rook_adjacency",
    "build_contiguity",
    "csr_from_geodataframe_contiguity",
]


def csr_from_pysal(W) -> tuple[list[int], list[int]]:
    """Build CSR (row_ptr, col_idx) from a libpysal.weights.W.

    Ensures symmetry (undirected) and no self loops. The node order follows W.id_order.
    """
    id_order = list(W.id_order)
    idx_of: dict = {id_: i for i, id_ in enumerate(id_order)}
    N = len(id_order)

    # Build adjacency sets for symmetry
    adj: list[set[int]] = [set() for _ in range(N)]
    for id_i in id_order:
        i = idx_of[id_i]
        for id_j in W.neighbors.get(id_i, []):
            j = idx_of.get(id_j)
            if j is None or i == j:
                continue
            adj[i].add(j)
            adj[j].add(i)

    row_ptr: list[int] = [0]
    col_idx: list[int] = []
    for i in range(N):
        col_idx.extend(sorted(adj[i]))
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def _dense_entry_is_binary(value: object) -> bool:
    """Return True when *value* is exactly the integer 0 or 1 (bool and 0.0/1.0 included)."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, np.integer)):
        return int(value) in (0, 1)
    if isinstance(value, (float, np.floating)):
        return float(value) in (0.0, 1.0)
    return False


def csr_from_dense(adj_mat: Iterable[Iterable[int]]) -> tuple[list[int], list[int]]:
    """Build CSR from a 0/1 dense adjacency matrix (square). Diagonal ignored.

    Every entry must be exactly ``0`` or ``1`` (Python or numpy integers,
    booleans, or the floats ``0.0``/``1.0``).  Weights, negative values, ``NaN``,
    strings and other objects are rejected with the offending ``(row, column)``
    positions, because the scan core treats adjacency as a pure relation and a
    weight it would silently discard is almost always a caller error (U-5).
    Diagonal entries must also be 0 or 1 but never become edges.

    Raises
    ------
    ValueError
        If the matrix is not square, contains an entry outside ``{0, 1}``, or is
        not symmetric (the C++ core requires an undirected graph with symmetric
        adjacency).
    """
    rows = [list(row) for row in adj_mat]
    N = len(rows)
    for i, row in enumerate(rows):
        if len(row) != N:
            raise ValueError(
                f"adj_mat must be square; row {i} has {len(row)} columns, expected {N}"
            )
    bad_positions = [
        (i, j)
        for i, row in enumerate(rows)
        for j, value in enumerate(row)
        if not _dense_entry_is_binary(value)
    ]
    if bad_positions:
        shown = ", ".join(f"({i}, {j})={rows[i][j]!r}" for i, j in bad_positions[:5])
        more = f" and {len(bad_positions) - 5} more" if len(bad_positions) > 5 else ""
        raise ValueError(
            "csr_from_dense requires a 0/1 adjacency matrix: every entry must be 0 or 1, "
            f"but {len(bad_positions)} entr{'y is' if len(bad_positions) == 1 else 'ies are'} "
            f"not: {shown}{more}. Weighted or non-numeric adjacency is not supported; "
            "binarise the matrix explicitly before calling csr_from_dense."
        )
    edges = [[bool(value) for value in row] for row in rows]
    # Symmetry check on the validated matrix: adj[i][j] must equal adj[j][i] for all i != j.
    for i in range(N):
        for j in range(i + 1, N):
            if edges[i][j] != edges[j][i]:
                raise ValueError(
                    f"adj_mat must be symmetric for an undirected graph: "
                    f"adj[{i}][{j}]={rows[i][j]} but adj[{j}][{i}]={rows[j][i]}"
                )
    row_ptr: list[int] = [0]
    col_idx: list[int] = []
    for r, flags in enumerate(edges):
        cols = [c for c, is_edge in enumerate(flags) if is_edge and c != r]
        col_idx.extend(cols)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def csr_knn_from_coords(
    coords,
    k: int,
    *,
    radius: float = float("inf"),
    include_self: bool = False,
    backend: str = "auto",
    metric: str = "euclidean",
) -> tuple[list[int], list[int]]:
    """Build a symmetric KNN CSR from coordinates using a KD-tree or BallTree.

    - coords: Nx2 numpy-like array
    - k: number of nearest neighbors per node (per direction)
    - radius: optional max distance cutoff; neighbors beyond this are dropped.
      For metric='euclidean': same units as coords.
      For metric='haversine': in radians (BallTree convention).
      Coordinates remain longitude/latitude in degrees; convert an angular
      radius from degrees to radians before passing it.
    - include_self: whether to include self-loops (defaults to False)
    - backend: 'auto' (prefer KDTree if SciPy available, else BallTree), 'kdtree', or 'balltree'.
      'balltree' needs scikit-learn (the optional ``flexscanx[knn]`` extra).
    - metric: 'euclidean' (default) or 'haversine' (lon/lat coords in degrees;
      radius must be in radians). 'haversine' needs scikit-learn (``flexscanx[knn]``).

    Returns (row_ptr, col_idx) for an undirected graph, neighbors sorted by index per row.
    """
    import numpy as _np

    X = _np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("coords must be Nx2 array")
    if not _np.all(_np.isfinite(X)):
        raise ValueError("coords must contain only finite numeric values")
    N = int(X.shape[0])
    if N == 0:
        raise ValueError("coords must contain at least one coordinate")
    if isinstance(k, (bool, _np.bool_)) or not isinstance(k, (int, _np.integer)):
        raise ValueError(f"k must be an integer, got {k!r}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > N:
        raise ValueError(f"k must be <= number of coordinates ({N}), got {k}")
    rad = float(radius)
    if _np.isnan(rad) or rad < 0.0:
        raise ValueError(f"radius must be non-negative or +inf, got {radius}")

    def _symmetric_csr(directed: list[list[int]]) -> tuple[list[int], list[int]]:
        adjacency = [set(row) for row in directed]
        for src, directed_neighbors in enumerate(directed):
            for dst in directed_neighbors:
                adjacency[dst].add(src)
        row_ptr = [0]
        col_idx: list[int] = []
        for adjacency_neighbors in adjacency:
            col_idx.extend(sorted(adjacency_neighbors))
            row_ptr.append(len(col_idx))
        return row_ptr, col_idx

    def _select_rows(
        initial_dists,
        initial_idxs,
        radius_query,
        exact_distances,
    ) -> tuple[list[int], list[int]]:
        initial_dists = _np.asarray(initial_dists).reshape(N, -1)
        initial_idxs = _np.asarray(initial_idxs).reshape(N, -1)
        directed: list[list[int]] = []
        for src in range(N):
            initial = [
                (float(distance), int(dst))
                for distance, dst in zip(initial_dists[src], initial_idxs[src])
                if 0 <= int(dst) < N and (include_self or int(dst) != src)
            ]
            initial.sort(key=lambda item: (item[0], item[1]))
            if not initial:
                directed.append([])
                continue

            boundary = initial[min(k, len(initial)) - 1][0]
            query_radius = min(boundary, rad)
            tolerance = max(1.0, abs(query_radius)) * 1e-12
            candidates = [int(value) for value in radius_query(src, query_radius + tolerance)]
            distances = exact_distances(src, candidates)
            ordered = sorted(
                (
                    (float(distance), dst)
                    for distance, dst in zip(distances, candidates)
                    if (include_self or dst != src) and float(distance) <= rad
                ),
                key=lambda item: (item[0], item[1]),
            )
            directed.append([dst for _, dst in ordered[:k]])
        return _symmetric_csr(directed)

    b = str(backend).strip().lower()
    m = str(metric).strip().lower()
    if b not in {"auto", "kdtree", "balltree"}:
        raise ValueError(
            f"backend must be one of {{'auto', 'kdtree', 'balltree'}}, got {backend!r}"
        )
    if m not in {"euclidean", "haversine"}:
        raise ValueError(f"metric must be one of {{'euclidean', 'haversine'}}, got {metric!r}")

    # Haversine path via scikit-learn BallTree (requires radians)
    if m == "haversine":
        if b == "kdtree":
            raise ValueError("haversine distance requires the BallTree backend")
        if _np.any(X[:, 0] < -180.0) or _np.any(X[:, 0] > 180.0):
            raise ValueError("haversine longitude values must be in [-180, 180] degrees")
        if _np.any(X[:, 1] < -90.0) or _np.any(X[:, 1] > 90.0):
            raise ValueError("haversine latitude values must be in [-90, 90] degrees")
        try:
            from sklearn.neighbors import BallTree  # type: ignore
        except ImportError as e:
            raise ImportError(
                "metric='haversine' requires scikit-learn's BallTree, which is not "
                "installed. Install it with `pip install flexscanx[knn]` (or "
                "`pip install scikit-learn`)."
            ) from e
        # Public coordinates are conventional (longitude, latitude), while
        # BallTree's haversine metric expects (latitude, longitude).
        Xrad = _np.radians(X[:, [1, 0]])
        tree = BallTree(Xrad, leaf_size=40, metric="haversine")
        qk = min(k + (0 if include_self else 1), N)
        dists, idxs = tree.query(Xrad, k=qk)

        def _query_radius_haversine(src: int, threshold: float):
            return tree.query_radius(Xrad[src : src + 1], r=threshold)[0]

        def _haversine_distances(src: int, candidates: list[int]):
            if not candidates:
                return _np.empty(0, dtype=float)
            lat1, lon1 = Xrad[src]
            lat2 = Xrad[candidates, 0]
            lon2 = Xrad[candidates, 1]
            sin_lat = _np.sin((lat2 - lat1) / 2.0)
            sin_lon = _np.sin((lon2 - lon1) / 2.0)
            a = sin_lat * sin_lat + _np.cos(lat1) * _np.cos(lat2) * sin_lon * sin_lon
            return 2.0 * _np.arcsin(_np.sqrt(_np.clip(a, 0.0, 1.0)))

        return _select_rows(dists, idxs, _query_radius_haversine, _haversine_distances)

    # Euclidean path via KDTree or BallTree
    if b in ("auto", "kdtree"):
        try:
            from scipy.spatial import cKDTree as _KDTree  # type: ignore

            tree = _KDTree(X)
            qk = min(k + (0 if include_self else 1), N)
            dists, idxs = tree.query(X, k=qk)

            def _query_radius_kdtree(src: int, threshold: float):
                return tree.query_ball_point(X[src], r=threshold)

            def _euclidean_distances_kdtree(src: int, candidates: list[int]):
                if not candidates:
                    return _np.empty(0, dtype=float)
                return _np.linalg.norm(X[candidates] - X[src], axis=1)

            return _select_rows(
                dists,
                idxs,
                _query_radius_kdtree,
                _euclidean_distances_kdtree,
            )
        except ImportError:
            if b == "kdtree":
                raise ImportError("KDTree backend requires SciPy (scipy.spatial.cKDTree)")
            # fall through to BallTree
    try:
        from sklearn.neighbors import BallTree  # type: ignore
    except ImportError as e:
        if b == "balltree":
            raise ImportError(
                "backend='balltree' requires scikit-learn, which is not installed. "
                "Install it with `pip install flexscanx[knn]` (or `pip install "
                "scikit-learn`), or use the default backend='auto' (SciPy KD-tree)."
            ) from e
        raise ImportError(
            "csr_knn_from_coords found neither SciPy's cKDTree nor scikit-learn's "
            "BallTree. SciPy is a declared runtime dependency (`pip install scipy`); "
            "scikit-learn is the optional `flexscanx[knn]` extra."
        ) from e
    tree = BallTree(X, leaf_size=40, metric="euclidean")
    qk = min(k + (0 if include_self else 1), N)
    dists, idxs = tree.query(X, k=qk)

    def _query_radius_balltree(src: int, threshold: float):
        return tree.query_radius(X[src : src + 1], r=threshold)[0]

    def _euclidean_distances_balltree(src: int, candidates: list[int]):
        if not candidates:
            return _np.empty(0, dtype=float)
        return _np.linalg.norm(X[candidates] - X[src], axis=1)

    return _select_rows(
        dists,
        idxs,
        _query_radius_balltree,
        _euclidean_distances_balltree,
    )


# Convenience wrappers for contiguity builders


def queen_adjacency(gdf, *, include_intersects: bool = False) -> tuple[list[int], list[int]]:
    """Return CSR for queen contiguity (touch at any point).

    When include_intersects=True, polygons whose interiors overlap are also treated
    as neighbours (useful for boundary files with slight overlaps).
    Output order follows gdf row order.
    """
    return csr_from_geodataframe_contiguity(
        gdf, method="queen", queen_include_intersects=bool(include_intersects)
    )


def rook_adjacency(gdf) -> tuple[list[int], list[int]]:
    """Return CSR for rook contiguity (shared boundary of non-zero length)."""
    return csr_from_geodataframe_contiguity(gdf, method="rook")


def build_contiguity(geodata_or_path, *, method: str = "queen", include_intersects: bool = False):
    """Build CSR contiguity from a path or a GeoDataFrame.

    - method: 'queen' or 'rook'
    - include_intersects: for queen, also treat overlapping polygons as neighbours
    """
    try:
        import geopandas as _gpd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("geopandas is required for build_contiguity") from e

    name = str(method).strip().lower()
    if name not in {"queen", "rook"}:
        raise ValueError(f"method must be one of {{'queen', 'rook'}}, got {method!r}")

    if isinstance(geodata_or_path, (str, os.PathLike)):
        gdf = _gpd.read_file(geodata_or_path)
    else:
        gdf = geodata_or_path
    if name == "rook":
        return rook_adjacency(gdf)
    return queen_adjacency(gdf, include_intersects=include_intersects)


# DE-9IM pattern for "shared boundary of non-zero length": interiors do not meet
# ("F" in the interior/interior cell) and the boundary/boundary intersection is
# one-dimensional ("1").  This is the rook predicate the docstrings state.
_ROOK_RELATE_PATTERN = "F***1****"


def csr_from_geodataframe_contiguity(
    gdf,
    *,
    method: str = "queen",
    queen_include_intersects: bool = False,
) -> tuple[list[int], list[int]]:
    """Build CSR (row_ptr, col_idx) from a GeoDataFrame polygon contiguity.

    - method: 'queen' (touch at any point) or 'rook' (shared boundary of non-zero length)
    - queen_include_intersects: when True, treat polygon overlaps (intersects) as
      adjacency in addition to touches. Useful for messy datasets where polygons
      slightly overlap; default False for strict queen='touches'.

    Both predicates are evaluated geometrically with shapely on every
    installation.  Until 2026-09 the function delegated to libpysal when it was
    importable, and libpysal decides contiguity by shared *vertices* (queen: one
    shared vertex; rook: one shared vertex pair), which disagrees with the
    documented predicates wherever polygons are not vertex-congruent (a vertex
    of one polygon lying inside an edge of another).  The graph, and therefore
    the reported cluster and p-value, then depended on whether an optional
    dependency was installed (FXA-164).  Requires shapely >= 2.0.
    The output order follows gdf row order.
    """
    try:
        import geopandas as _gpd  # noqa: F401
    except Exception as e:
        raise ImportError("geopandas is required for contiguity CSR from GeoDataFrame") from e
    try:
        import shapely as _shapely  # type: ignore[import-untyped]
        from shapely import STRtree as _STRtree  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - shapely is a hard requirement of geopandas
        raise ImportError("shapely >= 2.0 is required for contiguity CSR from GeoDataFrame") from e

    name = str(method).strip().lower()
    if name not in {"queen", "rook"}:
        raise ValueError(f"method must be one of {{'queen', 'rook'}}, got {method!r}")
    use_rook = name == "rook"

    for index, geometry in enumerate(gdf.geometry):
        if geometry is None or bool(getattr(geometry, "is_empty", False)):
            raise ValueError(f"geometry row {index} contains an empty geometry")
        if not bool(getattr(geometry, "is_valid", False)):
            raise ValueError(f"geometry row {index} contains an invalid geometry")

    geoms = np.asarray(gdf.geometry.values, dtype=object)
    N = int(len(geoms))
    adj: list[set[int]] = [set() for _ in range(N)]
    if N == 0:
        return [0], []

    tree = _STRtree(geoms)
    if use_rook:
        # Candidates touch; keep the pairs whose boundaries share a 1-D piece.
        left, right = tree.query(geoms, predicate="touches")
        keep = left != right
        left, right = left[keep], right[keep]
        if left.size:
            rook_ok = _shapely.relate_pattern(geoms[left], geoms[right], _ROOK_RELATE_PATTERN)
            left, right = left[rook_ok], right[rook_ok]
    else:
        predicate = "intersects" if bool(queen_include_intersects) else "touches"
        left, right = tree.query(geoms, predicate=predicate)
        keep = left != right
        left, right = left[keep], right[keep]

    for i, j in zip(left.tolist(), right.tolist()):
        adj[i].add(j)
        adj[j].add(i)
    row_ptr: list[int] = [0]
    col_idx: list[int] = []
    for i in range(N):
        col_idx.extend(int(neighbor) for neighbor in sorted(adj[i]))
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx
