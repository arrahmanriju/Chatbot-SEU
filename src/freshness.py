"""Evaluate only explicitly supplied source dates."""
from datetime import date


def potentially_outdated(updated_at, today=None, max_age_days=365):
    try:
        updated = date.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return True
    age = ((today or date.today()) - updated).days
    return age < 0 or age > max_age_days
