"""SamplingMixin — adds sampling methods to FeatherStore."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from featherstore.sampling import bootstrap_sample, sample_fraction, sample_n


class SamplingMixin:
    """Mixin that exposes sampling helpers on FeatherStore instances."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        group: str,
        frac: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        stratify_col: Optional[str] = None,
        replace: bool = False,
    ) -> pd.DataFrame:
        """Load *group* and return a random sample.

        Exactly one of *frac* or *n* must be provided.
        """
        if (frac is None) == (n is None):
            raise ValueError("Provide exactly one of 'frac' or 'n'.")

        df: pd.DataFrame = self.load(group)  # type: ignore[attr-defined]

        if frac is not None:
            return sample_fraction(df, frac=frac, seed=seed, stratify_col=stratify_col)
        return sample_n(df, n=n, seed=seed, replace=replace)  # type: ignore[arg-type]

    def bootstrap(
        self,
        group: str,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load *group* and return a bootstrap (with-replacement) sample."""
        df: pd.DataFrame = self.load(group)  # type: ignore[attr-defined]
        return bootstrap_sample(df, n=n, seed=seed)
