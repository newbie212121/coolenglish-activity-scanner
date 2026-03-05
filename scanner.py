#!/usr/bin/env python3
"""
Cool English — Activity Request Scanner
----------------------------------------
Connects to Zoho Mail via IMAP, finds [Activity Request] emails,
clusters them with Claude AI, and saves results to requests.json.

Setup:
  pip install anthropic python-dotenv

Environment variables (.env or GitHub Secrets):
  ZOHO_EMAIL       your Zoho email address
  ZOHO_PASSWORD    Zoho app-specific password (not your main password)
                   → Generate at: https://accounts.zoho.com/home#security/app-passwords
  ANTHROPIC_API_KEY  your Anthropic API key

Run:
  python scanner.py
  python scanner.py --days 7     # scan last 7 days (default: 1)
  python scanner.py --output results.json
"""

import imaplib
import email
import json
import re
import argparse
import os
from datetime import datetime, timedelta
from email.header import decode_header
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("❌ Missing dependency: pip install anthropic")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional, can use real env vars


# ─── Configuration ────────────────────────────────────────────────────────────

ZOHO_IMAP_HOST = "imap.zoho.com"
ZOHO_IMAP_PORT = 993
SUBJECT_FILTER = "[Activity Request]"
OUTPUT_FILE = "requests.json"


# ─── Email Fetching ───────────────────────────────────────────────────────────

def connect_zoho(email_addr: str, password: str) -> imaplib.IMAP4_SSL:
    print(f"🔌 Connecting to Zoho IMAP as {email_addr}...")
    mail = imaplib.IMAP4_SSL(ZOHO_IMAP_HOST, ZOHO_IMAP_PORT)
    mail.login(email_addr, password)
    mail.select("INBOX")
    print("✅ Connected.")
    return mail


def fetch_activity_requests(mail: imaplib.IMAP4_SSL, days: int = 1) -> list[dict]:
    since_date = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
    search_query = f'(SINCE "{since_date}" SUBJECT "{SUBJECT_FILTER}")'

    print(f"🔍 Searching for emails since {since_date}...")
    _, message_ids = mail.search(None, search_query)

    ids = message_ids[0].split()
    print(f"📬 Found {len(ids)} activity request email(s).")

    requests = []
    for msg_id in ids:
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        subject = _decode_header(msg["Subject"])
        date_str = msg["Date"]
        sender_name = _decode_header(msg.get("From", ""))
        body = _extract_body(msg)

        idea = _parse_idea_from_body(body)
        teacher_email = _extract_teacher_email(body, sender_name)

        if idea:
            requests.append({
                "subject": subject,
                "teacher": teacher_email,
                "idea": idea,
                "date": _parse_date(date_str),
                "raw_body": body[:500],
            })
            print(f"  → \"{idea}\" from {teacher_email}")

    return requests


def _decode_header(value: str) -> str:
    if not value:
        return ""
    decoded_parts = decode_header(value)
    parts = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            parts.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(part)
    return " ".join(parts)


def _extract_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode("utf-8", errors="replace")
    else:
        body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
    return body


def _parse_idea_from_body(body: str) -> str | None:
    """Extract the activity idea from the Cool English email template."""
    # Look for the pattern after "Activity Request from [email]:"
    patterns = [
        r"Activity Request from .+?:\s*\n+(.+?)(?:\n|$)",
        r"Activity Idea:\s*\n+Activity Request from .+?:\s*\n+(.+?)(?:\n|$)",
        r"idea[:\s]+(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
        if match:
            idea = match.group(1).strip()
            if 3 < len(idea) < 300:
                return idea
    return None


def _extract_teacher_email(body: str, sender: str) -> str:
    """Pull teacher email from body or fall back to sender."""
    match = re.search(r"From:\s*([\w.+-]+@[\w.-]+)", body)
    if match:
        return match.group(1)
    match = re.search(r"[\w.+-]+@[\w.-]+", sender)
    if match:
        return match.group(0)
    return sender


def _parse_date(date_str: str) -> str:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


# ─── AI Clustering ────────────────────────────────────────────────────────────

def cluster_with_claude(requests: list[dict], existing: list[dict] = None) -> list[dict]:
    """Send all requests to Claude and get back clustered/ranked results."""
    if not requests:
        return []

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    all_requests = (existing or []) + requests
    idea_list = "\n".join(
        f'{i+1}. "{r["idea"]}" (from {r["teacher"]}, {r["date"]})'
        for i, r in enumerate(all_requests)
    )

    print(f"\n🤖 Sending {len(all_requests)} requests to Claude for clustering...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing ESL teacher activity requests for a language learning website called Cool English.

Here are ALL submitted activity requests:
{idea_list}

Group these into thematic clusters. For each cluster:
- Give a short, clear theme name (e.g. "Medical / Doctor Visits")
- Pick a relevant emoji
- Count how many requests fit
- List 2-3 short example phrases from the actual requests
- List the request numbers (1-based) that belong to this cluster
- List the unique teachers who requested this

Return ONLY a JSON array, no markdown, no explanation:
[
  {{
    "theme": "Medical / Doctor Visits",
    "emoji": "🏥",
    "count": 3,
    "examples": ["Going to the doctor", "Describing symptoms"],
    "requestIds": [1, 2, 15],
    "teachers": ["teacher@gmail.com"]
  }}
]
Sort by count descending. Every request must appear in exactly one cluster."""
        }]
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    clusters = json.loads(text)
    print(f"✅ Got {len(clusters)} clusters from Claude.")
    return clusters


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_existing(output_file: str) -> dict:
    path = Path(output_file)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"requests": [], "clusters": [], "last_updated": None}


def save_results(output_file: str, all_requests: list[dict], clusters: list[dict]):
    data = {
        "last_updated": datetime.now().isoformat(),
        "total": len(all_requests),
        "clusters": clusters,
        "requests": all_requests,
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {output_file}")
    print(f"   {len(all_requests)} requests · {len(clusters)} clusters")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cool English Activity Request Scanner")
    parser.add_argument("--days", type=int, default=1, help="Days to scan back (default: 1)")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output JSON file")
    parser.add_argument("--recluster", action="store_true", help="Re-cluster all existing requests too")
    args = parser.parse_args()

    # Check env vars
    for var in ["ZOHO_EMAIL", "ZOHO_PASSWORD", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(var):
            print(f"❌ Missing environment variable: {var}")
            exit(1)

    # Load existing data
    existing_data = load_existing(args.output)
    existing_requests = existing_data.get("requests", [])
    print(f"📂 Loaded {len(existing_requests)} existing requests from {args.output}")

    # Fetch new emails
    mail = connect_zoho(os.environ["ZOHO_EMAIL"], os.environ["ZOHO_PASSWORD"])
    new_requests = fetch_activity_requests(mail, days=args.days)
    mail.logout()

    if not new_requests and not args.recluster:
        print("\n✨ No new requests found. Nothing to update.")
        return

    # Merge (deduplicate by idea+teacher)
    existing_keys = {(r["idea"], r["teacher"]) for r in existing_requests}
    unique_new = [r for r in new_requests if (r["idea"], r["teacher"]) not in existing_keys]
    print(f"\n➕ {len(unique_new)} new unique requests to add.")

    all_requests = existing_requests + unique_new

    # Cluster everything
    clusters = cluster_with_claude(unique_new if not args.recluster else [], 
                                   existing_requests if not args.recluster else all_requests)

    if args.recluster:
        clusters = cluster_with_claude(all_requests, [])

    # Save
    save_results(args.output, all_requests, clusters)

    # Print summary
    print("\n📊 Top Requested Activities:")
    for i, c in enumerate(clusters[:5], 1):
        print(f"  #{i} {c['emoji']} {c['theme']} — {c['count']} request(s)")


if __name__ == "__main__":
    main()
