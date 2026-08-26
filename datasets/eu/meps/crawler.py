import time
from dataclasses import dataclass, field
from typing import Any

from banal import ensure_list
from requests.exceptions import HTTPError
from rigour.urls import build_url

from zavod import Context
from zavod import helpers as h
from zavod.entity import Entity
from zavod.runtime.http_ import request_hash
from zavod.stateful.positions import PositionCategorisation, categorise

# One term roster fits in a single response; this bounds it and flags overflow.
ROSTER_LIMIT = 10000


@dataclass
class Term:
    number: int
    start: str | None
    end: str | None


@dataclass
class OrgInfo:
    local_id: str
    name: str | None
    acronym: str | None
    countries: list[str] = field(default_factory=list)


class LeakyBucketRateLimiter:
    """Leaky-bucket rate limiter: paces calls to `rate` per second by sleeping."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._tat = time.monotonic()

    def acquire(self) -> None:
        now = time.monotonic()
        self._tat = max(self._tat, now)
        if self._tat > now:
            time.sleep(self._tat - now)
        self._tat += self._interval


# The API allows 500 requests per 5 minutes (~1.67/s); 1.5/s keeps a safety margin.
rate_limiter = LeakyBucketRateLimiter(1.5)


def last_segment(value: Any) -> str | None:
    """Return the last path segment of an EU authority URI, e.g.
    `.../country/BEL` -> `BEL`, `.../human-sex/MALE` -> `MALE`."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value:
        return None
    return value.rsplit("/", 1)[-1]


def pick_label(value: Any) -> str | None:
    """Return a label: an API string, or English (else any) from a language-keyed dict."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    strings = [v for v in value.values() if isinstance(v, str)]
    return (
        value["en"] if isinstance(value.get("en"), str) else next(iter(strings), None)
    )


def classification(membership: dict[str, Any]) -> str | None:
    """Return a membership's entity type, e.g. `EU_POLITICAL_GROUP`."""
    return last_segment(membership.get("membershipClassification"))


def fetch_data(
    context: Context, path: str, cache_days: int = 7, **params: Any
) -> list[Any]:
    """Fetch a JSON-LD endpoint and return its `data` array."""
    params.setdefault("format", "application/ld+json")
    url = f"https://data.europarl.europa.eu/api/v2{path}"
    # Recreate fetch_json's cache fingerprint so we only pace on a real network call.
    fingerprint = request_hash(build_url(url, params))
    if context.cache.get(fingerprint, max_age=cache_days) is None:
        rate_limiter.acquire()
    body = context.fetch_json(url, params=params, cache_days=cache_days)
    return ensure_list(body.get("data"))


def fetch_terms(context: Context) -> list[Term]:
    """Walk the parliament institution bodies `org/ep-{n}` to learn each term's
    date range. Stop at the first term that does not exist."""
    terms: list[Term] = []
    # 16 is far above any real term number; the loop stops at the first 404.
    for number in range(1, 16):
        try:
            rows = fetch_data(context, f"/corporate-bodies/ep-{number}")
        except HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                break
            raise
        temporal = rows[0].get("temporal") if rows else None
        temporal = temporal if isinstance(temporal, dict) else {}
        terms.append(Term(number, temporal.get("startDate"), temporal.get("endDate")))
    return terms


def fetch_org(context: Context, org_ref: str, cache: dict[str, OrgInfo]) -> OrgInfo:
    """Resolve an `org/{id}` reference to its name, acronym and countries."""
    if org_ref in cache:
        return cache[org_ref]
    local_id = org_ref.split("/", 1)[-1]
    rows = fetch_data(context, f"/corporate-bodies/{local_id}")
    data = rows[0] if rows else {}
    countries = [c for c in map(last_segment, ensure_list(data.get("represents"))) if c]
    # prefLabel is the body's full name; altLabel is a shorter form, used as fallback.
    name = pick_label(data.get("prefLabel")) or pick_label(data.get("altLabel"))
    acronym = data.get("label") if isinstance(data.get("label"), str) else None
    # The API uses "-" as a placeholder for a missing value.
    name = None if name == "-" else name
    acronym = None if acronym == "-" else acronym
    info = OrgInfo(local_id=local_id, name=name, acronym=acronym, countries=countries)
    cache[org_ref] = info
    return info


def crawl_group_membership(
    context: Context,
    person: Entity,
    membership: dict[str, Any],
    is_eu_group: bool,
    cache: dict[str, OrgInfo],
) -> None:
    """Emit the political group or national party and the person's membership in it."""
    org_ref = membership.get("organization")
    if not isinstance(org_ref, str):
        return
    info = fetch_org(context, org_ref, cache)

    org = context.make("Organization")
    # The API models a group as a distinct body per term. Key by name so the
    # same party or group is one entity across terms, not one per term.
    org.id = context.make_slug(
        "eu-group" if is_eu_group else "nat-party", info.name or info.local_id
    )
    org.add("name", info.name)
    org.add("name", info.acronym)
    if is_eu_group:
        org.add("country", "eu")
    else:
        org.add("country", info.countries)
    context.emit(org)

    # memberDuring carries the start and end dates of the membership period.
    period = membership.get("memberDuring")
    period = period if isinstance(period, dict) else {}
    entity = context.make("Membership")
    entity.id = context.make_id(
        person.id, org.id, period.get("startDate"), period.get("endDate")
    )
    entity.add("member", person)
    entity.add("organization", org)
    role = last_segment(membership.get("role"))
    if role is not None:
        entity.add("role", role.replace("_", " ").lower())
    h.apply_date(entity, "startDate", period.get("startDate"))
    h.apply_date(entity, "endDate", period.get("endDate"))
    context.emit(entity)


def crawl_mep(
    context: Context,
    mep_id: str,
    position: Entity,
    categorisation: PositionCategorisation,
    cache: dict[str, OrgInfo],
) -> None:
    """Fetch one MEP and emit the person, their mandates and group memberships."""
    rows = fetch_data(context, f"/meps/{mep_id}")
    if not rows:
        context.log.warning("No data for MEP", mep_id=mep_id)
        return
    data = rows[0]

    # Names come as plain strings, but gender and citizenship as EU authority URIs.
    person = context.make("Person")
    person.id = context.make_slug(data["identifier"])
    person.add("name", pick_label(data.get("label")))
    person.add("firstName", data.get("givenName"))
    person.add("lastName", data.get("familyName"))
    person.add("gender", last_segment(data.get("hasGender")))
    person.add("birthPlace", data.get("placeOfBirth"))
    for citizenship in ensure_list(data.get("citizenship")):
        person.add("citizenship", last_segment(citizenship))
    h.apply_date(person, "birthDate", data.get("bday"))
    h.apply_date(person, "deathDate", data.get("deathDate"))
    person.add("sourceUrl", f"https://www.europarl.europa.eu/meps/en/{mep_id}")

    memberships = ensure_list(data.get("hasMembership"))

    # A person is emitted only if they hold an MEP mandate that is still
    # relevant for PEP screening. `make_occupancy` decides relevance from the
    # mandate end date and drops the ones that ended too long ago.
    # Each term served is a separate mandate membership in `org/ep-{term}`.
    occupancies: list[Entity] = []
    for membership in memberships:
        if classification(membership) != "EU_INSTITUTION":
            continue
        org_ref = membership.get("organization")
        if not isinstance(org_ref, str) or not org_ref.startswith("org/ep-"):
            continue
        period = membership.get("memberDuring")
        period = period if isinstance(period, dict) else {}
        occupancy = h.make_occupancy(
            context,
            person,
            position,
            start_date=period.get("startDate"),
            end_date=period.get("endDate"),
            categorisation=categorisation,
        )
        if occupancy is not None:
            occupancies.append(occupancy)
    if not occupancies:
        return

    for occupancy in occupancies:
        context.emit(occupancy)
    context.emit(person)

    for membership in memberships:
        group = classification(membership)
        if group == "EU_POLITICAL_GROUP":
            crawl_group_membership(context, person, membership, True, cache)
        elif group == "NATIONAL_POLITICAL_GROUP":
            crawl_group_membership(context, person, membership, False, cache)


def crawl(context: Context) -> None:
    """Crawl MEPs from every parliamentary term within the PEP relevance window."""
    position = h.make_position(
        context,
        "Member of the European Parliament",
        wikidata_id="Q27169",
        country="eu",
        topics=["gov.igo", "gov.legislative"],
        lang="eng",
    )
    categorisation = categorise(context, position, default_is_pep=True)
    context.emit(position)

    # Keep terms that are ongoing or ended within the PEP relevance window.
    cutoff = h.earliest_term_start(position.get("topics"))
    terms = [t for t in fetch_terms(context) if t.end is None or t.end >= cutoff]
    context.log.info(
        "Crawling MEP terms within PEP relevance window",
        cutoff=cutoff,
        terms=[t.number for t in terms],
    )

    mep_ids: set[str] = set()
    for term in terms:
        rows = fetch_data(
            context,
            "/meps",
            **{"parliamentary-term": term.number, "limit": ROSTER_LIMIT},
        )
        if len(rows) >= ROSTER_LIMIT:
            context.log.warning("Term roster may be truncated", term=term.number)
        for row in rows:
            mep_ids.add(str(row["identifier"]))
    context.log.info("Fetched MEP roster", count=len(mep_ids))

    cache: dict[str, OrgInfo] = {}
    for mep_id in sorted(mep_ids):
        crawl_mep(context, mep_id, position, categorisation, cache)
