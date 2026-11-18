"""Merge approved public SEU records and update the existing FAISS artifact."""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from src.loader import DEFAULT_DATA_PATH, OFFICIAL_UPDATES_PATH, VERIFIED_STUDENT_PATH, read_records
from src.store import INDEX_FILE, build_vectorstore, index_directory, load_index, read_manifest

DEFAULT_INPUT = Path(__file__).with_name("approved_seu_sources.jsonl")
AGENT = "Chatbot-SEU-knowledge-updater"
MAX_DOWNLOAD = 10 * 1024 * 1024
FIELDS = ("id", "source_url", "title", "clean_text")


def official_url(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    return (parsed.scheme == "https" and not parsed.username and not parsed.password
            and parsed.port in (None, 443) and (host == "seu.edu.bd" or host.endswith(".seu.edu.bd")))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class PageText(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}

    def __init__(self):
        super().__init__()
        self.hidden = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID and (self.hidden or tag in self.SKIP):
            self.hidden.append(tag)
        elif tag in {"p", "h1", "h2", "h3", "li", "tr", "br", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.hidden:
            while self.hidden.pop() != tag:
                pass
        elif not self.hidden and tag in {"p", "li", "tr", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip() + " ")


def read_public(opener, url, limit=MAX_DOWNLOAD):
    time.sleep(1)
    request = Request(url, headers={"User-Agent": AGENT})
    with opener.open(request, timeout=18) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Approved document exceeds the download limit.")
        return data, response.headers.get_content_type(), response.headers.get_content_charset()


def collect(url):
    """Fetch a single approved URL, respecting robots rules and failing on redirects."""
    if not official_url(url):
        raise ValueError("Expected an official SEU HTTPS URL.")
    host = urlsplit(url).netloc
    opener = build_opener(NoRedirect())
    try:
        rules, _, _ = read_public(opener, f"https://{host}/robots.txt", 1024 * 1024)
    except HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        robots = RobotFileParser()
        robots.parse(rules.decode("utf-8", errors="replace").splitlines())
        if not robots.can_fetch(AGENT, url):
            raise ValueError("The publisher disallows collection for this URL.")
        delay = robots.crawl_delay(AGENT) or robots.crawl_delay("*") or 0
        if delay > 1:
            time.sleep(delay - 1)
        rate = robots.request_rate(AGENT) or robots.request_rate("*")
        if rate and rate.requests and rate.seconds / rate.requests > max(delay, 1):
            time.sleep(rate.seconds / rate.requests - max(delay, 1))
    raw, media_type, charset = read_public(opener, url)
    if media_type == "application/pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    elif media_type == "text/html":
        parser = PageText()
        parser.feed(raw.decode(charset or "utf-8", errors="replace"))
        text = "".join(parser.parts)
    elif media_type == "text/plain":
        text = raw.decode(charset or "utf-8", errors="replace")
    else:
        raise ValueError("Unsupported content type.")
    return clean_text(text)


def clean_text(text):
    return "\n".join(line for row in text.splitlines() if (line := " ".join(row.split())))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_key(record):
    return digest(" ".join(record["clean_text"].split()).casefold()), record.get("program"), (
        record.get("academic_session") or record.get("admission_session"))


def approved_record(item, fetch, applying):
    if not isinstance(item, dict):
        raise ValueError("Source manifest row must be an object.")
    row = dict(item)
    if not official_url(row.get("source_url", "")):
        raise ValueError("Expected an official SEU HTTPS URL.")
    download = row.pop("fetch", False)
    expected = row.pop("expected_sha256", None)
    if download:
        if not fetch:
            raise ValueError("A fetch row requires --fetch.")
        text = collect(row["source_url"])
        if applying and expected != digest(text):
            raise ValueError("Downloaded text requires a matching reviewed expected_sha256.")
        row["clean_text"] = text
        row["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    elif not isinstance(row.get("clean_text"), str):
        raise ValueError("Provide reviewed clean_text or an approved fetch row.")
    row["clean_text"] = clean_text(row["clean_text"])
    scope = [row["source_url"], row.get("category"), row.get("program"),
             row.get("academic_session") or row.get("admission_session")]
    row.setdefault("id", "seu-public-" + digest(json.dumps(scope, ensure_ascii=False))[:20])
    for key in FIELDS:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing or invalid {key}.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,140}", row["id"]):
        raise ValueError("Invalid stable document ID.")
    if len(row["clean_text"]) < 40:
        raise ValueError("No usable document content.")
    for key in ("category", "program", "academic_session", "effective_date", "document_type"):
        if row.get(key) is not None and not isinstance(row[key], str):
            raise ValueError(f"{key} must be text.")
    if row.get("effective_date"):
        date.fromisoformat(row["effective_date"])
    if row.get("retrieved_at"):
        if datetime.fromisoformat(row["retrieved_at"]).tzinfo is None:
            raise ValueError("retrieved_at must include a timezone.")
    if (not isinstance(row.get("supersedes", []), list)
            or any(type(value) not in (str, int) for value in row.get("supersedes", []))):
        raise ValueError("supersedes must be a list of document IDs.")
    row["verification"] = "official_public"
    return row


def external_fields(row):
    fields = {key: value for key, value in row.items() if key not in {"retrieved_at", "supersedes"}}
    history_prefix = row["id"][:100] + "--old-"
    links = sorted(str(link) for link in row.get("supersedes", ()) if not str(link).startswith(history_prefix))
    return fields, links


def equivalent(first, second):
    return external_fields(first) == external_fields(second)


def merge(existing, immutable, incoming):
    result = list(existing)
    protected = {str(row["id"]): row for row in immutable}
    active = {str(row["id"]): index for index, row in enumerate(result)}
    superseded = {str(key) for row in (*immutable, *existing) for key in row.get("supersedes", ())}
    outcomes = []
    for row in incoming:
        key = row["id"]
        if key in protected:
            outcomes.append((key, "failed", "Use a new ID and supersedes to revise an immutable source."))
            continue
        position = active.get(key)
        if position is not None and equivalent(row, result[position]):
            outcomes.append((key, "skipped", "Unchanged."))
            continue
        previous_links = {str(value) for value in result[position].get("supersedes", ())} if position is not None else set()
        links = {str(value) for value in row.get("supersedes", ())} | previous_links
        if key in superseded or (links - previous_links) & superseded or key in links or not links <= protected.keys() | active.keys():
            outcomes.append((key, "failed", "Supersession must reference an existing active document."))
            continue
        if any(content_key(old) == content_key(row) for old in (*immutable, *result)
               if str(old["id"]) != key and str(old["id"]) not in links):
            outcomes.append((key, "skipped", "Duplicate text in the same program/session scope."))
            continue
        if position is not None:
            previous = result[position]
            revision = f"{key[:100]}--old-{digest(json.dumps(previous, sort_keys=True))[:12]}"
            if revision not in active:
                active[revision] = len(result)
                result.append({**previous, "id": revision, "superseded_by": key})
            row = {**row, "supersedes": sorted(links | {revision})}
            result[position] = row
        else:
            active[key] = len(result)
            result.append(row)
        superseded.update(str(value) for value in row.get("supersedes", ()))
        outcomes.append((key, "updated" if position is not None else "added", sorted(row.get("supersedes", []))))
    return result, outcomes


def update_index(rows, overlay, index_dir, original):
    index_dir.mkdir(parents=True, exist_ok=True)
    lock = index_dir / ".knowledge-update.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        actual = overlay.read_bytes() if overlay.exists() else None
        if actual != original:
            raise ValueError("The overlay changed during review. Rerun the update.")
        with tempfile.TemporaryDirectory(dir=overlay.parent, prefix=".seu-update-") as temp:
            staged = Path(temp)
            prepared = staged / "official_updates.jsonl"
            prepared.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            live = index_dir / INDEX_FILE
            backup = index_dir.parent / "kb-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup.mkdir(parents=True)
            previous = None
            if live.exists():
                shutil.copy2(live, backup / INDEX_FILE)
                previous = read_manifest(index_dir)
            overlay_existed = overlay.exists()
            if overlay_existed:
                shutil.copy2(overlay, backup / overlay.name)

            build_vectorstore(staged, updates_path=prepared, append_from=index_dir)
            verified = read_manifest(staged)
            new_ids = {str(r["metadata"].get("id")) for r in verified["documents"]}
            expected_ids = {str(row["id"]) for row in rows}
            if previous:
                expected_ids.update(str(r["metadata"].get("id")) for r in previous["documents"])
                if previous["model"] != verified["model"]:
                    raise ValueError("The updated index must use the existing embedding model.")
            if not expected_ids <= new_ids:
                raise ValueError(f"Missing expected source IDs: {sorted(expected_ids - new_ids)}")
            load_index(staged)
            with tempfile.NamedTemporaryFile(dir=index_dir, delete=False) as fp:
                candidate = Path(fp.name)
            try:
                shutil.copy2(staged / INDEX_FILE, candidate)
                os.replace(prepared, overlay)
                try:
                    os.replace(candidate, live)
                except OSError:
                    if overlay_existed:
                        shutil.copy2(backup / overlay.name, prepared)
                        os.replace(prepared, overlay)
                    else:
                        overlay.unlink()
                    raise
            finally:
                candidate.unlink(missing_ok=True)
            return backup, len(verified["documents"])
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fetch", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview changes without writing (default).")
    mode.add_argument("--apply", action="store_true", help="Update the overlay and existing index.")
    args = parser.parse_args()
    original = OFFICIAL_UPDATES_PATH.read_bytes() if OFFICIAL_UPDATES_PATH.exists() else None
    existing = [json.loads(line) for line in original.decode("utf-8-sig").splitlines()
                if line.strip()] if original is not None else []
    immutable = [*read_records(DEFAULT_DATA_PATH), *read_records(VERIFIED_STUDENT_PATH)]
    accepted, outcomes = [], []
    for position, item in enumerate(read_records(args.input), start=1):
        try:
            row = approved_record(item, args.fetch, args.apply)
            accepted.append(row)
            if item.get("fetch") and not args.apply:
                print(json.dumps({"id": row["id"], "sha256": digest(row["clean_text"]),
                                  "review_text": row["clean_text"]}, ensure_ascii=False))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            outcomes.append((str(position), "failed", str(exc)))
    merged, results = merge(existing, immutable, accepted)
    outcomes.extend(results)
    counts = Counter(outcome for _, outcome, _ in outcomes)
    for row in outcomes:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({key: counts[key] for key in ("added", "updated", "skipped", "failed")}))
    if counts["failed"]:
        return 1
    if args.apply and (counts["added"] or counts["updated"] or not (index_directory() / INDEX_FILE).exists()):
        backup, chunks = update_index(merged, OFFICIAL_UPDATES_PATH, index_directory(), original)
        print(json.dumps({"backup": str(backup), "chunks": chunks, "index": str(index_directory())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
