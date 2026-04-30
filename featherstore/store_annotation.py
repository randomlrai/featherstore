"""AnnotationMixin — integrates annotation support into FeatherStore."""

from typing import Dict, Optional
from featherstore.annotation import (
    set_annotation,
    get_annotation,
    remove_annotation,
    list_annotations,
)


class AnnotationMixin:
    """Mixin that adds annotation methods to FeatherStore."""

    def annotate(self, group: str, note: str, author: Optional[str] = None) -> Dict:
        """Attach a note to *group*, optionally recording the *author*.

        Overwrites any existing annotation for that group.
        Returns the stored annotation entry.
        """
        return set_annotation(self.store_path, group, note, author=author)

    def get_annotation(self, group: str) -> Optional[Dict]:
        """Return the annotation dict for *group*, or ``None`` if not set."""
        return get_annotation(self.store_path, group)

    def remove_annotation(self, group: str) -> bool:
        """Remove the annotation for *group*.

        Returns ``True`` if an annotation existed and was removed,
        ``False`` if the group had no annotation.
        """
        return remove_annotation(self.store_path, group)

    def list_annotations(self) -> Dict[str, Dict]:
        """Return a mapping of group name → annotation entry for all annotated groups."""
        return list_annotations(self.store_path)
