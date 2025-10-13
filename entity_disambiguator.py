"""
Entity disambiguation and canonicalisation.

After extraction, entity names may be ambiguous.  For example, the
string "Princeton" could refer to a university or a town, and
different forms such as "A. Einstein" and "Albert Einstein" should be
linked to the same entity.  This module provides a simple
disambiguator that attempts to resolve names against the Wikidata
knowledge base.  In cases where no external resolution is possible
(e.g. offline operation), the disambiguator maintains an internal
canonical mapping to ensure that repeated references are normalised.

The resolution strategy is as follows:

1. If a name has already been seen, reuse the previously assigned
   canonical identifier.
2. Otherwise, query the Wikidata API to search for entities with
   matching labels.  If a result is returned, use the Wikidata ID
   (e.g. ``Q937``).  Store this mapping for future occurrences.
3. If Wikidata lookup fails (network issues or no results), normalise
   the name by lowercasing and replacing spaces with underscores.

This simple approach can be improved by incorporating context from the
surrounding text or by integrating with a larger entity linking
pipeline.  The class is designed to be replaced with a more
sophisticated implementation without affecting the rest of the system.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional

import requests

from .entity_extractor import Triple

logger = logging.getLogger(__name__)


class EntityDisambiguator:
    """Resolve entity names to canonical identifiers using Wikidata.

    Parameters
    ----------
    language: str, optional
        The language code used when querying Wikidata.  Defaults to
        ``"en"``.
    limit: int, optional
        Maximum number of search results to consider.  Defaults to 1.
    session: requests.Session, optional
        Optional HTTP session to reuse connections.  If not provided
        a new session is created for each request.
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    def __init__(self, language: str = "en", limit: int = 1, session: Optional[requests.Session] = None) -> None:
        self.language = language
        self.limit = limit
        self.session = session or requests.Session()
        # Internal cache mapping raw names to canonical identifiers
        self._cache: Dict[str, str] = {}

    def disambiguate(self, triples: Iterable[Triple]) -> List[Triple]:
        """Return a new list of triples with canonicalised entity names.

        Parameters
        ----------
        triples:
            An iterable of ``Triple`` objects.  The ``subject`` and
            ``object`` fields are examined.

        Returns
        -------
        List[Triple]
            A new list of triples with ``subject`` and ``object``
            replaced by canonical identifiers.
        """
        resolved: List[Triple] = []
        for triple in triples:
            subj_id = self._resolve_entity(triple.subject)
            obj_id = self._resolve_entity(triple.object)
            resolved.append(
                Triple(
                    subject=subj_id,
                    predicate=triple.predicate,
                    object=obj_id,
                    metadata=triple.metadata,
                )
            )
        return resolved

    def _resolve_entity(self, name: str) -> str:
        """Resolve an entity name to a canonical identifier.

        If the name has been seen before, return the cached value.
        Otherwise, attempt to query Wikidata.  On failure, return a
        normalised version of the name itself.
        """
        key = name.strip()
        if key in self._cache:
            return self._cache[key]
        try:
            params = {
                'action': 'wbsearchentities',
                'language': self.language,
                'format': 'json',
                'search': key,
                'limit': self.limit,
            }
            response = self.session.get(self.WIKIDATA_API, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get('search'):
                entity_id = data['search'][0].get('id')
                if entity_id:
                    self._cache[key] = entity_id
                    return entity_id
        except Exception as e:
            logger.debug("Wikidata lookup failed for %s: %s", name, e)
        # Fallback: normalise the name (lowercase, replace spaces)
        normalised = re.sub(r'\s+', '_', key.strip().lower())
        self._cache[key] = normalised
        return normalised