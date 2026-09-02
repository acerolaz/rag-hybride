def resolve_ingest_action(existing_hash: str | None, new_hash: str) -> str:
    if existing_hash is None:
        return "created"
    if existing_hash == new_hash:
        return "unchanged"
    return "updated"
