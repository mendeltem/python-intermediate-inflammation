import os
import numpy as np
from pathlib import Path
from typing import List, Sequence, Optional


def filter_list(
    files: Sequence[str],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return a list of file names that contain **all** substrings in *include* and
    **none** of the substrings in *exclude*.

    Parameters
    ----------
    files : Sequence[str]
        The collection of file names (or full paths) to filter.
    include : Sequence[str] | None, default None
        Substrings that **must** be present in the file name.  If *None* or an empty
        sequence, no positive filtering is applied.
    exclude : Sequence[str] | None, default None
        Substrings that **must not** be present in the file name.  If *None* or an
        empty sequence, no negative filtering is applied.

    Returns
    -------
    list[str]
        The filtered list of file names in the same order as supplied.
    """

    include = list(include or [])
    exclude = list(exclude or [])

    def _is_wanted(name: str) -> bool:
        return all(term in name for term in include) and not any(term in name for term in exclude)

    return [f for f in files if _is_wanted(f)]



def find_single_file(
    files: Sequence[str],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> str:
    """Return **exactly one** file that satisfies the *include*/*exclude* criteria.

    Raises
    ------
    FileNotFoundError
        If no file matches the criteria.
    ValueError
        If more than one file matches the criteria.
    """

    matches = filter_list(files, include, exclude)

    if len(matches) == 0:
        raise FileNotFoundError("No file matches the given criteria.")
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one match, but found {len(matches)}: {matches}")

    return matches[0]





