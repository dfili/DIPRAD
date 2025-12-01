#!/usr/bin/env python3
"""
fetch_htb.py — modular pipeline with rich logging and tqdm

Features:
- Uses long-lived 6-month token (no refresh)
- Async fetching: teams → team members → user profiles
- JSONL raw saving and Parquet conversion
- Checkpointing for resuming user profile fetches
- Rich logging (timestamp, level, module, line)
- tqdm progress bars for all stages
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

import httpx
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# ============================================================
# 1. CONFIG & ENV
# ============================================================
load_dotenv()
API_BASE = "https://labs.hackthebox.com/api/v4"
TEAMS_ENDPOINT = f"{API_BASE}/rankings/teams"
TEAM_MEMBERS_ENDPOINT = lambda tid: f"{API_BASE}/team/members/{tid}"
USER_PROFILE_ENDPOINT = lambda uid: f"{API_BASE}/profile/progress/challenges/{uid}"

HTB_TOKEN = os.getenv("HTB_API_TOKEN")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_BASE = 1.0
MAX_CONCURRENCY = 5

# ============================================================
# 2. LOGGING SETUP (rich style)
# ============================================================
logger = logging.getLogger("htb_fetch")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s.%(msecs)03d [%(levelname)s] (%(funcName)s:%(lineno)d) → %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(DATA_DIR / "fetch.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# ============================================================
# 3. TOKEN MANAGER
# ============================================================
class TokenManager:
    def __init__(self, token: str):
        if not token:
            logger.error("HTB_API_TOKEN not found in .env")
            raise ValueError("HTB_API_TOKEN is required")
        self.token = token

    def get_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/json, */*"}

# ============================================================
# 4. ASYNC HTTP CLIENT WITH RETRY
# ============================================================
class AsyncHTTP:
    def __init__(self, token_mgr: TokenManager, concurrency: int = MAX_CONCURRENCY):
        self.token_mgr = token_mgr
        self.client = httpx.AsyncClient(headers=self.token_mgr.get_header())
        self.sem = asyncio.Semaphore(concurrency)

    async def rate_limited(self):
        async with self.sem:
            await asyncio.sleep(0.2)

    async def close(self):
        await self.client.aclose()
    

    async def get(self, url: str) -> Any:
        last_exc = None
        for attempt in range(1, DEFAULT_RETRIES + 1):
            try:
                async with self.sem:
                    await asyncio.sleep(0.2)
                    resp = await self.client.get(url, timeout=20)
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after is not None:
                            delay = int(retry_after)
                        else:
                            delay = min(5, attempt)  # small exponential fallback

                        logger.warning(f"429 Too Many Requests on {url}. Retrying in {delay}s…")
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                last_exc = e
                backoff = DEFAULT_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(f"{url} failed (attempt {attempt}) → {repr(e)}. Retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
        logger.error(f"Giving up on {url} after {DEFAULT_RETRIES} attempts → {repr(last_exc)}")
        return None

# ============================================================
# 5. FETCH FUNCTIONS
# ============================================================
async def fetch_teams(http: AsyncHTTP) -> List[Dict[str, Any]]:
    logger.info("Fetching teams…")

    resp = await http.get(TEAMS_ENDPOINT)
    if not resp:
        logger.error("Teams endpoint returned no data")
        return []
    if isinstance(resp, dict) and resp.get("status") and isinstance(resp.get("data"), list):
        teams = resp["data"]
    else:
        logger.warning("Unexpected teams format; using raw response")
        teams = resp if isinstance(resp, list) else []

    logger.info(f"Fetched {len(teams)} teams")
    return teams

async def fetch_team_members(http: AsyncHTTP, team_id: str) -> List[Dict[str, Any]]:
    url = TEAM_MEMBERS_ENDPOINT(team_id)
    data = await http.get(url) or []
    for m in data:
        m['team_id'] = team_id
    logger.info(f"Fetched {len(data)} members for team {team_id}")
    return data

async def fetch_user_profile(http: AsyncHTTP, user_id: str) -> Dict[str, Any]:
    url = USER_PROFILE_ENDPOINT(user_id)
    data = await http.get(url)
    if data:
        logger.debug(f"Fetched profile for user {user_id}")
        return {"user_id": user_id, "profile": data.get("profile", data)}
    return {}

# ============================================================
# 6. PARQUET SERIALIZATION
# ============================================================
def save_parquet(data: List[Dict], out_path: Path):
    if not data:
        logger.warning(f"No data to save to {out_path}")
        return
    df = pd.DataFrame(data)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")

# ============================================================
# 7. MAIN PIPELINE
# ============================================================
async def main():
    token_mgr = TokenManager(HTB_TOKEN)
    http = AsyncHTTP(token_mgr)
    try:
        # 1) Teams
        # teams = await fetch_teams(http)
        # (RAW_DIR / "teams.jsonl").write_text('\n'.join(json.dumps(t) for t in teams))
        # save_parquet(teams, PARQUET_DIR / "teams.parquet")
        # teams = pd.read_parquet(PARQUET_DIR / "teams.parquet").to_dict(orient="records")
        
        # 2) Team Members
        # team_members = []
        # for team in tqdm(teams, desc="Teams → Members"):
        #     members = await fetch_team_members(http, str(team.get("id")))
        #     team_members.extend(members)
        team_members = pd.read_json(RAW_DIR / "team_members.jsonl", lines=True).to_dict(orient="records")
        # (RAW_DIR / "team_members.jsonl").write_text('\n'.join(json.dumps(m) for m in team_members))
        # save_parquet(team_members, PARQUET_DIR / "team_members.parquet")
        
        # 3) User Profiles with checkpointing
        checkpoint_file = CHECKPOINT_DIR / "fetched_profiles.txt"
        fetched_already = set(checkpoint_file.read_text().splitlines()) if checkpoint_file.exists() else set()

        profiles = []
        for user in tqdm(team_members, desc="Users → Profiles"):
            if user.get("public") == 1:
                uid = str(user.get("id"))
                if uid in fetched_already:
                    continue
                profile = await fetch_user_profile(http, uid)
                if profile:
                    profiles.append(profile)
                    with checkpoint_file.open("a") as cp:
                        cp.write(uid + "\n")

        (RAW_DIR / "user_profiles.jsonl").write_text('\n'.join(json.dumps(p) for p in profiles))
        save_parquet(profiles, PARQUET_DIR / "user_profiles.parquet")

        logger.info("Pipeline complete.")
    finally:
        await http.close()

if __name__ == "__main__":
    asyncio.run(main())