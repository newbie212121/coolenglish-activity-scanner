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

# Emails from these addresses are skipped entirely (your own accounts)
OWN_ADDRESSES = {
    "john@coolenglish.net",
    "jt2128@gmail.com",
}

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
        from_addr  = msg.get("fromAddress", "").lower().strip()
        subject    = msg.get("subject", "")

        # Skip reply threads — these are your responses back to teachers
        if re.match(r"^re:", subject, re.IGNORECASE):
            print(f"  ⏭️  Skipping reply thread: {subject[:60]}")
            skipped_spam += 1
            continue

        # Skip emails sent from your own addresses
        if from_addr in OWN_ADDRESSES:
            print(f"  ⏭️  Skipping own-address email from {from_addr}")
            skipped_spam += 1
            continue
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

def _safe_json_parse(text, label="response"):
    """Parse JSON, salvaging partial output if truncated."""
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error in {label}: {e}. Attempting salvage...")
        last_close = text.rfind("},")
        if last_close == -1:
            last_close = text.rfind("}")
        if last_close != -1:
            salvaged = text[:last_close + 1] + "]"
            try:
                result = json.loads(salvaged)
                print(f"♻️  Salvaged {len(result)} items.")
                return result
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse {label}:\n{text[:500]}")


def cluster_all_with_claude(all_requests):
    """Full re-cluster of every request. Used for --recluster flag."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not all_requests:
        return []

    idea_list = "\n".join(
        f'{i+1}. "{r["idea"]}" (from {r["teacher"]})'
        for i, r in enumerate(all_requests)
    )
    print(f"\n🤖 Full re-clustering {len(all_requests)} requests with Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You are analyzing ESL teacher activity requests for Cool English.

Here are ALL submitted activity requests:
{idea_list}

Group into thematic clusters relevant to ESL teaching. Ignore or group separately any that look like tests, spam, or support emails.

Return ONLY a JSON array, no markdown, no trailing commas:
[{{"theme":"Grammar - Verb Tenses","emoji":"⏰","count":3,"examples":["present perfect","simple past"],"requestIds":[1,2,3],"teachers":["teacher@example.com"]}}]

Sort by count descending. Keep examples brief (under 40 chars each).
"""}]
    )

    clusters = _safe_json_parse(response.content[0].text, "full re-cluster")
    print(f"✅ {len(clusters)} clusters.")
    return clusters


def cluster_new_with_claude(new_requests, existing_clusters, existing_count):
    """
    Incremental clustering: only send NEW requests to Claude.
    Claude assigns each to an existing cluster or creates a new one.
    Old requests are never re-sent — existing clusters are updated in-place.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not new_requests:
        return existing_clusters

    # New requests get IDs continuing from where existing ones left off
    idea_list = "\n".join(
        f'{existing_count + i + 1}. "{r["idea"]}" (from {r["teacher"]})'
        for i, r in enumerate(new_requests)
    )

    cluster_summary = "\n".join(
        f'- "{c["theme"]}" {c["emoji"]}'
        for c in existing_clusters
    ) or "(none yet — create clusters from scratch)"

    print(f"\n🤖 Incrementally clustering {len(new_requests)} new request(s) with Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are classifying new ESL activity requests for Cool English.

EXISTING clusters (reuse these theme names exactly when a good match exists):
{cluster_summary}

NEW requests to classify:
{idea_list}

For each new request, assign it to the best existing cluster OR create a new one.
Skip anything that is spam, a test, a support email, or not an ESL activity request.

Return ONLY a JSON array, no markdown:
[
  {{"requestId": 5, "theme": "Grammar - Verb Tenses", "isNew": false}},
  {{"requestId": 6, "theme": "Pronunciation", "emoji": "🗣️", "isNew": true}}
]

Rules:
- isNew=false → use an exact theme name from the EXISTING list above
- isNew=true → provide a brand new theme name AND an emoji
- Omit spam/test/support requests entirely
"""}]
    )

    assignments = _safe_json_parse(response.content[0].text, "incremental cluster")

    # Mutable dict of existing clusters keyed by theme
    clusters = {c["theme"]: dict(c) for c in existing_clusters}

    for a in assignments:
        req_idx = a["requestId"] - existing_count - 1  # 0-based index into new_requests
        if req_idx < 0 or req_idx >= len(new_requests):
            print(f"  ⚠️  Assignment requestId {a['requestId']} out of range, skipping.")
            continue
        req = new_requests[req_idx]
        theme = a["theme"]

        if a.get("isNew"):
            clusters[theme] = {
                "theme": theme,
                "emoji": a.get("emoji", "❓"),
                "count": 1,
                "examples": [req["idea"][:50]],
                "requestIds": [a["requestId"]],
                "teachers": [req["teacher"]],
            }
            print(f"  🆕 New cluster: {theme}")
        elif theme in clusters:
            c = clusters[theme]
            c["count"] += 1
            c["requestIds"].append(a["requestId"])
            if req["teacher"] not in c["teachers"]:
                c["teachers"].append(req["teacher"])
            if len(c.get("examples", [])) < 3:
                c["examples"].append(req["idea"][:50])
            print(f"  ➕ Added to '{theme}'")
        else:
            print(f"  ⚠️  Unknown theme '{theme}' — skipping.")

    result = sorted(clusters.values(), key=lambda x: x["count"], reverse=True)
    print(f"✅ {len(result)} clusters total.")
    return result


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

    if args.recluster:
        # Full re-cluster: send everything, rebuild clusters from scratch
        clusters = cluster_all_with_claude(all_requests)
    else:
        # Incremental: only classify new requests, merge into cached clusters
        existing_clusters = existing_data.get("clusters", [])
        clusters = cluster_new_with_claude(unique_new, existing_clusters, len(existing_requests))

    save_results(args.output, all_requests, clusters)

    print("\n📊 Top Requested Activities:")
    for i, c in enumerate(clusters[:5], 1):
        print(f"  #{i} {c['emoji']} {c['theme']} — {c['count']}x")


if __name__ == "__main__":
    main()
