import hashlib

from moh.core.populations import rank_key


def heuristic_rank(item):
    return rank_key(item.evaluation.status, item.evaluation.utility, item.heuristic.id)


def _ranked(members):
    if len({x.heuristic.id for x in members}) != len(members):
        raise ValueError("duplicate candidate IDs")
    return sorted(members, key=heuristic_rank)


def select_parents(members, policy, count, rng):
    ranked = _ranked(members)
    if type(count) is not int or not 1 <= count <= len(ranked):
        raise ValueError("insufficient parents")
    if policy == "best":
        return tuple(ranked[:count])
    if policy == "random":
        return tuple(rng.sample(ranked, count))
    if policy != "tournament":
        raise ValueError("invalid parent policy")
    parents = []
    for _ in range(count):
        chosen = min(rng.sample(ranked, min(2, len(ranked))), key=heuristic_rank)
        parents.append(chosen)
        ranked.remove(chosen)
    return tuple(parents)


def select_survivors(members, capacity, policy):
    ranked = _ranked(members)
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("invalid capacity")
    if policy == "elitist":
        return tuple(ranked[:capacity])
    if policy != "diversity":
        raise ValueError("invalid survivor policy")
    seen, unique, duplicates = set(), [], []
    # Failures must never displace successful duplicates for source diversity.
    for status in ("success", "failed"):
        for item in ranked:
            if item.evaluation.status != status:
                continue
            digest = hashlib.sha256(item.heuristic.source_code.encode()).digest()
            if digest in seen:
                duplicates.append(item)
            else:
                unique.append(item)
                seen.add(digest)
        if status == "success":
            unique.extend(duplicates)
            duplicates.clear()
    return tuple(sorted((unique + duplicates)[:capacity], key=heuristic_rank))
