#!/usr/bin/env python3
"""
Cool English — Activity Request Scanner (Zoho REST API)
Includes spam/test filtering and AI clustering.

Environment variables (GitHub Secrets):
  ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN, ANTHROPIC_API_KEY
"""

import requests
import json
import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("❌ pip install anthropic requests")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Config ───────────────────────────────────────────────────────────────────

SUBJECT_FILTER = "[Activity Request]"
OUTPUT_FILE = "requests.json"
ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
ZOHO_API_BASE = "https://mail.zoho.com/api"

# Phrases that indicate a test/spam submission — filtered out automatically
SPAM_PATTERNS = [
    r"^test\b", r"^testing\b", r"^hello\b", r"^hi\b", r"^asdf",
    r"^[a-z]{1,4}$",          # single short random words
    r"^\d+$",                  # only numbers
    r"spam", r"ignore this",
    r"^\.+$",                  # just dots
]
SPAM_RE = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)

MIN_IDEA_LENGTH = 8   # anything shorter is likely garbage


# ─── OAuth ────────────────────────────────────────────────────────────────────

def get_access_token():
    print("🔐 Getting Zoho access token...")
    resp = requests.post(ZOHO_TOKEN_URL, data={
        "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"],
        "client_id":     os.environ["ZOHO_CLIENT_ID"],
        "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
        "grant_type":    "refresh_token",
    })
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError(f"No access token: {resp.json()}")
    print("✅ Token obtained.")
    return token


def get_account_id(token):
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    resp = requests.get(f"{ZOHO_API_BASE}/accounts", headers=headers)
    resp.raise_for_status()
    accounts = resp.json().get("data", [])
    if not accounts:
        raise ValueError("No accounts found.")
    aid = str(accounts[0]["accountId"])
    print(f"📬 Account: {accounts[0].get('emailAddress')} ({aid})")
    return aid


# ─── Fetch + Filter ───────────────────────────────────────────────────────────

def fetch_activity_requests(token, account_id, days=1):
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    since_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    print(f"🔍 Searching last {days} days for '{SUBJECT_FILTER}'...")

    all_messages = []
    start = 1
    while True:
        params = {
            "searchKey": f'subject:"{SUBJECT_FILTER}"',
            "receivedTime": since_ms,
            "start": start,
            "limit": 50,
            "includeto": "true",
        }
        resp = requests.get(
            f"{ZOHO_API_BASE}/accounts/{account_id}/messages/search",
            headers=headers, params=params
        )
        resp.raise_for_status()
        msgs = resp.json().get("data", [])
        if not msgs:
            break
        all_messages.extend(msgs)
        if len(msgs) < 50:
            break
        start += 50
        time.sleep(0.3)

    print(f"📨 Found {len(all_messages)} raw emails.")

    results = []
    skipped_spam = 0

    for msg in all_messages:
        message_id = msg.get("messageId")
        folder_id  = msg.get("folderId")
        date_ms    = msg.get("receivedtime", 0)
        date_str   = datetime.fromtimestamp(date_ms / 1000).strftime("%Y-%m-%d") if date_ms else "unknown"
        from_addr  = msg.get("fromAddress", "")

        idea = None
        teacher_email = from_addr

        try:
            body_resp = requests.get(
                f"{ZOHO_API_BASE}/accounts/{account_id}/folders/{folder_id}/messages/{message_id}/content",
                headers=headers,
            )
            if body_resp.status_code == 200:
                raw = body_resp.json().get("data", {}).get("content", "")
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"\s+", " ", text).strip()
                idea = _parse_idea(text)
                teacher_email = _extract_email(text) or from_addr
        except Exception as e:
            print(f"  ⚠️ Body error {message_id}: {e}")

        if not idea:
            idea = msg.get("summary", "").strip()

        if not idea or len(idea) < MIN_IDEA_LENGTH:
            skipped_spam += 1
            continue

        if SPAM_RE.search(idea):
            print(f"  🗑️  Filtered spam: \"{idea}\"")
            skipped_spam += 1
            continue

        results.append({
            "teacher": teacher_email,
            "idea": idea,
            "date": date_str,
        })
        print(f"  ✅ \"{idea}\" — {teacher_email}")

    print(f"\n📊 Kept {len(results)} real requests, filtered {skipped_spam} spam/test submissions.")
    return results


def _parse_idea(body):
    patterns = [
        r"Activity Request from .+?:\s*(.+?)(?:Submitted|$)",
        r"Activity Idea[:\s]+.+?:\s*(.+?)(?:Submitted|$)",
    ]
    for p in patterns:
        m = re.search(p, body, re.IGNORECASE)
        if m:
            idea = m.group(1).strip()
            if MIN_IDEA_LENGTH < len(idea) < 300:
                return idea
    return None


def _extract_email(body):
    for prefix in ["From:", "Email:"]:
        m = re.search(rf"{prefix}\s*([\w.+-]+@[\w.-]+)", body)
        if m:
            return m.group(1)
    return None


# ─── Claude Clustering ────────────────────────────────────────────────────────

def cluster_with_claude(requests_list, existing=None):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    all_reqs = (existing or []) + requests_list
    if not all_reqs:
        return []

    idea_list = "\n".join(
        f'{i+1}. "{r["idea"]}" (from {r["teacher"]}, {r["date"]})'
        for i, r in enumerate(all_reqs)
    )

    print(f"\n🤖 Clustering {len(all_reqs)} requests with Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You are analyzing ESL teacher activity requests for a language learning website called Cool English.

Here are ALL submitted activity requests:
{idea_list}

Group into thematic clusters relevant to ESL teaching. Ignore or group separately any that look like tests or spam.

For each cluster return:
- theme: clear ESL topic name (e.g. "Medical / Doctor Visits")
- emoji: relevant emoji
- count: number of requests
- examples: 2-3 short example phrases from actual requests
- requestIds: 1-based list of request numbers
- teachers: unique teacher emails

Return ONLY a JSON array, no markdown, no trailing commas:
[{{"theme":"...","emoji":"...","count":3,"examples":["..."],"requestIds":[1,2],"teachers":["..."]}}]

Sort by count descending. Be concise in examples to keep the response short.
"""}]
    )

    text = response.content[0].text.strip().replace("```json","").replace("```","").strip()

    # If response was truncated, attempt to salvage valid clusters up to the break point
    try:
        clusters = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error: {e}. Attempting to salvage partial response...")
        # Truncate to last complete object and close the array
        last_close = text.rfind("},")
        if last_close == -1:
            last_close = text.rfind("}")
        if last_close != -1:
            salvaged = text[:last_close + 1] + "]"
            try:
                clusters = json.loads(salvaged)
                print(f"♻️  Salvaged {len(clusters)} clusters from partial response.")
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse Claude's response even after salvage attempt.\nRaw text:\n{text[:500]}")
        else:
            raise ValueError(f"Claude returned unparseable response:\n{text[:500]}")

    print(f"✅ {len(clusters)} clusters.")
    return clusters


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_existing(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"requests": [], "clusters": [], "last_updated": None}


def save_results(path, all_requests, clusters):
    data = {
        "last_updated": datetime.now().isoformat(),
        "total": len(all_requests),
        "clusters": clusters,
        "requests": all_requests,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(all_requests)} requests · {len(clusters)} clusters → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--recluster", action="store_true")
    args = parser.parse_args()

    for var in ["ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(var):
            print(f"❌ Missing: {var}")
            exit(1)

    existing_data = load_existing(args.output)
    existing_requests = existing_data.get("requests", [])
    print(f"📂 Loaded {len(existing_requests)} existing requests.")

    token = get_access_token()
    account_id = get_account_id(token)
    new_requests = fetch_activity_requests(token, account_id, days=args.days)

    existing_keys = {(r["idea"], r["teacher"]) for r in existing_requests}
    unique_new = [r for r in new_requests if (r["idea"], r["teacher"]) not in existing_keys]
    print(f"➕ {len(unique_new)} new unique requests.")

    all_requests = existing_requests + unique_new

    if not unique_new and not args.recluster:
        print("✨ Nothing new.")
        return

    clusters = cluster_with_claude(
        all_requests if args.recluster else unique_new,
        [] if args.recluster else existing_requests
    )

    save_results(args.output, all_requests, clusters)

    print("\n📊 Top Requested Activities:")
    for i, c in enumerate(clusters[:5], 1):
        print(f"  #{i} {c['emoji']} {c['theme']} — {c['count']}x")


if __name__ == "__main__":
    main()
