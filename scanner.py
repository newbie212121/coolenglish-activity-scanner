#!/usr/bin/env python3
"""
Cool English — Activity Request Scanner (Zoho REST API version)
----------------------------------------------------------------
Uses Zoho Mail REST API (free plan compatible) instead of IMAP.
Requires OAuth 2.0 tokens — see README for one-time setup.

Setup:
  pip install anthropic requests python-dotenv

Environment variables (.env or GitHub Secrets):
  ZOHO_CLIENT_ID       from api-console.zoho.com
  ZOHO_CLIENT_SECRET   from api-console.zoho.com
  ZOHO_REFRESH_TOKEN   generated once via OAuth flow
  ANTHROPIC_API_KEY    your Anthropic API key

Run:
  python scanner.py
  python scanner.py --days 14    # scan last 14 days
  python scanner.py --recluster  # re-cluster all existing requests
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
    print("❌ Missing dependency: pip install anthropic requests")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ────────────────────────────────────────────────────────────

SUBJECT_FILTER = "[Activity Request]"
OUTPUT_FILE = "requests.json"
ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
ZOHO_API_BASE = "https://mail.zoho.com/api"


# ─── OAuth ────────────────────────────────────────────────────────────────────

def get_access_token() -> str:
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
        print(f"❌ Token response: {resp.json()}")
        raise ValueError("Could not get access token. Check your ZOHO_REFRESH_TOKEN.")
    print("✅ Access token obtained.")
    return token


def get_account_id(token: str) -> str:
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    resp = requests.get(f"{ZOHO_API_BASE}/accounts", headers=headers)
    resp.raise_for_status()
    accounts = resp.json().get("data", [])
    if not accounts:
        raise ValueError("No Zoho Mail accounts found.")
    account_id = str(accounts[0]["accountId"])
    email = accounts[0].get("emailAddress", "unknown")
    print(f"📬 Using account: {email} (ID: {account_id})")
    return account_id


# ─── Fetch Emails ─────────────────────────────────────────────────────────────

def fetch_activity_requests(token: str, account_id: str, days: int = 1) -> list:
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    since_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"🔍 Searching for '{SUBJECT_FILTER}' emails (last {days} days)...")

    all_messages = []
    start = 1
    limit = 50

    while True:
        params = {
            "searchKey": f'subject:"{SUBJECT_FILTER}"',
            "receivedTime": since_ms,
            "start": start,
            "limit": limit,
            "includeto": "true",
        }
        resp = requests.get(
            f"{ZOHO_API_BASE}/accounts/{account_id}/messages/search",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        messages = resp.json().get("data", [])
        if not messages:
            break
        all_messages.extend(messages)
        print(f"  📨 Fetched {len(all_messages)} emails so far...")
        if len(messages) < limit:
            break
        start += limit
        time.sleep(0.3)

    print(f"📬 Found {len(all_messages)} activity request email(s).")

    results = []
    for msg in all_messages:
        message_id = msg.get("messageId")
        folder_id = msg.get("folderId")
        subject = msg.get("subject", "")
        date_ms = msg.get("receivedtime", 0)
        date_str = datetime.fromtimestamp(date_ms / 1000).strftime("%Y-%m-%d") if date_ms else "unknown"
        from_addr = msg.get("fromAddress", "")

        idea = None
        teacher_email = from_addr

        try:
            body_resp = requests.get(
                f"{ZOHO_API_BASE}/accounts/{account_id}/folders/{folder_id}/messages/{message_id}/content",
                headers=headers,
            )
            if body_resp.status_code == 200:
                body_data = body_resp.json().get("data", {})
                body_text = body_data.get("content", "")
                body_text = re.sub(r"<[^>]+>", " ", body_text)
                body_text = re.sub(r"\s+", " ", body_text).strip()
                idea = _parse_idea(body_text)
                teacher_email = _extract_email(body_text) or from_addr
        except Exception as e:
            print(f"  ⚠️ Body fetch error for {message_id}: {e}")

        if not idea:
            idea = msg.get("summary", "").strip()

        if idea and len(idea) > 3:
            results.append({
                "subject": subject,
                "teacher": teacher_email,
                "idea": idea,
                "date": date_str,
            })
            print(f"  → \"{idea}\" from {teacher_email}")

    return results


def _parse_idea(body: str):
    patterns = [
        r"Activity Request from .+?:\s*(.+?)(?:Submitted|$)",
        r"Activity Idea[:\s]+.+?:\s*(.+?)(?:Submitted|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            idea = match.group(1).strip()
            if 3 < len(idea) < 300:
                return idea
    return None


def _extract_email(body: str):
    for prefix in ["From:", "Email:"]:
        match = re.search(rf"{prefix}\s*([\w.+-]+@[\w.-]+)", body)
        if match:
            return match.group(1)
    return None


# ─── Claude Clustering ────────────────────────────────────────────────────────

def cluster_with_claude(new_requests: list, existing: list = None) -> list:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    all_requests = (existing or []) + new_requests

    if not all_requests:
        return []

    idea_list = "\n".join(
        f'{i+1}. "{r["idea"]}" (from {r["teacher"]}, {r["date"]})'
        for i, r in enumerate(all_requests)
    )

    print(f"\n🤖 Sending {len(all_requests)} requests to Claude for clustering...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": f"""You are analyzing ESL teacher activity requests for a language learning website called Cool English.

Here are ALL submitted activity requests:
{idea_list}

Group into thematic clusters. For each cluster return:
- theme: short clear name (e.g. "Medical / Doctor Visits")
- emoji: relevant emoji
- count: number of requests
- examples: 2-3 short phrases from actual requests
- requestIds: 1-based list of request numbers in this cluster
- teachers: unique teacher emails

Return ONLY a JSON array, no markdown:
[{{"theme":"...","emoji":"...","count":3,"examples":["..."],"requestIds":[1,2],"teachers":["..."]}}]

Sort by count descending. Every request must be in exactly one cluster."""}]
    )

    text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    clusters = json.loads(text)
    print(f"✅ Got {len(clusters)} clusters.")
    return clusters


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_existing(output_file: str) -> dict:
    path = Path(output_file)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"requests": [], "clusters": [], "last_updated": None}


def save_results(output_file: str, all_requests: list, clusters: list):
    data = {
        "last_updated": datetime.now().isoformat(),
        "total": len(all_requests),
        "clusters": clusters,
        "requests": all_requests,
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(all_requests)} requests · {len(clusters)} clusters → {output_file}")


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
    print(f"➕ {len(unique_new)} new unique requests to add.")

    all_requests = existing_requests + unique_new

    if not unique_new and not args.recluster:
        print("✨ Nothing new. Done.")
        return

    clusters = cluster_with_claude(
        all_requests if args.recluster else unique_new,
        [] if args.recluster else existing_requests
    )

    save_results(args.output, all_requests, clusters)

    print("\n📊 Top Requested Activities:")
    for i, c in enumerate(clusters[:5], 1):
        print(f"  #{i} {c['emoji']} {c['theme']} — {c['count']} request(s)")


if __name__ == "__main__":
    main()
