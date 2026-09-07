"""Seam for future integration with external services.

Backcountry Steward doesn't have a public API yet. Once it does, implement
submit_to_backcountry_steward below to POST the per-event species counts.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def submit_to_backcountry_steward(events) -> None:
    logger.info(
        "Backcountry Steward integration not yet configured; skipping submission "
        "of %d events.",
        len(events),
    )
