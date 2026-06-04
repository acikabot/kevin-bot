"""
Meet Kevin YouTube Summarizer Bot
Runs on GitHub Actions — completely free

Every hour it checks Meet Kevin's YouTube channel for new videos.
Only processes videos published on or after June 4th 2026.
When a new video is found it:
  1. Fetches the full transcript
  2. Sends it to Groq AI for a deep summary
  3. Emails you the summary
  4. Sends a push notification via ntfy

Tracks already-processed videos in seen_videos.json committed back to the repo.

Run modes:
  python kevin_bot.py           — normal hourly run
  python kevin_bot.py test      — test mode: summarizes latest video only, no seen list update
"""

import os
import sys
import json
import re
import smtplib
import requests
import feedparser
from openai import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
EMAIL_SENDER    = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD  = os.environ["EMAIL_PASSWORD"]
EMAIL_RECIPIENT = os.environ["EMAIL_RECIPIENT"]
KEVIN_NTFY      = os.environ.get("KEVIN_NTFY", "")

# Only summarize videos published on or after this date
CUTOFF_DATE = datetime(2026, 6, 4, tzinfo=timezone.utc)

# Meet Kevin's YouTube channel
CHANNEL_ID  = "UCUvvj5lwue7PspotMDjk5UA"
CHANNEL_RSS = f"https://www.youtube.com/feeds/videos.xml?channel_id={CHANNEL_ID}"

SEEN_FILE = "seen_videos.json"

# Run mode — "test" summarizes latest video only, skips seen list
TEST_MODE = len(sys.argv) > 1 and sys.argv[1] == "test"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ─── Seen videos tracker ──────────────────────────────────────────────────────

def load_seen_videos() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("seen", []))
    return set()

def save_seen_videos(seen: set):
    with open(SEEN_FILE, "w") as f:
        json.dump({
            "seen": list(seen),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }, f, indent=2)

# ─── YouTube RSS feed ─────────────────────────────────────────────────────────

def parse_published_date(entry) -> datetime:
    """Parse the published date from an RSS entry. Returns UTC datetime."""
    # Try the published_parsed field first (feedparser auto-parses it)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        import time
        ts = time.mktime(entry.published_parsed)
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    # Fallback: parse the published string manually
    published_str = entry.get("published", "")
    if published_str:
        try:
            return parsedate_to_datetime(published_str).astimezone(timezone.utc)
        except Exception:
            pass

    # If we can't parse the date, assume it's new to be safe
    return datetime.now(timezone.utc)

def get_latest_videos(max_videos=5):
    """
    Fetch latest videos from Meet Kevin's channel via RSS.
    Returns list of dicts: id, title, link, published, published_dt
    """
    feed = feedparser.parse(CHANNEL_RSS)
    videos = []
    for entry in feed.entries[:max_videos]:
        video_id = entry.get("yt_videoid", "")
        if not video_id:
            raw_id = entry.get("id", "")
            video_id = raw_id.split(":")[-1] if ":" in raw_id else ""

        if video_id:
            published_dt = parse_published_date(entry)
            videos.append({
                "id":           video_id,
                "title":        entry.get("title", "Unknown Title"),
                "link":         f"https://www.youtube.com/watch?v={video_id}",
                "published":    entry.get("published", ""),
                "published_dt": published_dt,
            })
    return videos

# ─── Transcript fetcher ───────────────────────────────────────────────────────

def get_transcript(video_id: str) -> str:
    """Fetch full transcript using the new youtube-transcript-api v1.x API."""
    try:
        # New API: YouTubeTranscriptApi().fetch() handles language fallback automatically
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])
        full_text = " ".join(seg.text for seg in fetched)
        full_text = re.sub(r"\[.*?\]", "", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return full_text

    except TranscriptsDisabled:
        print(f"  Transcripts disabled for {video_id}")
        return None
    except NoTranscriptFound:
        print(f"  No English transcript for {video_id}")
        return None
    except Exception as e:
        print(f"  Transcript error for {video_id}: {e}")
        return None

# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_transcript(transcript: str, max_chars=12000) -> list:
    """Split long transcripts into chunks at sentence boundaries."""
    if len(transcript) <= max_chars:
        return [transcript]

    chunks = []
    while len(transcript) > max_chars:
        cut = transcript[:max_chars].rfind(". ")
        cut = (cut + 1) if cut != -1 else max_chars
        chunks.append(transcript[:cut].strip())
        transcript = transcript[cut:].strip()
    if transcript:
        chunks.append(transcript)
    return chunks

# ─── Summarizer ───────────────────────────────────────────────────────────────

def summarize_transcript(transcript: str, video_title: str, video_url: str) -> str:
    chunks = chunk_transcript(transcript)
    if len(chunks) == 1:
        return _summarize_full(chunks[0], video_title, video_url)
    else:
        print(f"  Long video — processing in {len(chunks)} chunks...")
        partials = []
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...")
            partials.append(_summarize_partial(chunk, video_title, i+1, len(chunks)))
        return _combine_summaries(partials, video_title, video_url)

def _summarize_full(transcript: str, title: str, url: str) -> str:
    prompt = f"""You are summarizing a financial YouTube video by Meet Kevin (Kevin Paffrath), a well-known US financial creator covering stocks, real estate, the economy, and markets.

VIDEO TITLE: {title}
VIDEO URL: {url}

FULL TRANSCRIPT:
{transcript}

Write a detailed structured summary with exactly these sections:

🎯 MAIN THESIS
Kevin's central argument in 1-2 sentences.

📌 KEY POINTS
The 5-8 most important points Kevin makes. Each point gets 2-3 sentences — include his reasoning and any data or evidence cited.

📈 STOCKS & TICKERS MENTIONED
Every stock, ETF, or asset mentioned. For each: what Kevin said about it, his stance (bullish/bearish/neutral), and why. Write "None" if no tickers mentioned.

💡 ACTIONABLE TAKEAWAYS
3-5 concrete things Kevin suggests viewers do or watch for.

⚠️ RISKS OR CONCERNS MENTIONED
2-4 risks or warnings Kevin raised.

🔮 FORWARD-LOOKING STATEMENTS
Any predictions or upcoming catalysts Kevin mentioned.

📊 SENTIMENT
One word (Bullish / Bearish / Mixed) followed by one sentence explaining the overall tone.

Be direct and specific. No filler. This summary replaces watching the video."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def _summarize_partial(transcript: str, title: str, chunk_num: int, total: int) -> str:
    prompt = f"""This is part {chunk_num} of {total} of a Meet Kevin financial YouTube video titled: "{title}"

TRANSCRIPT SECTION:
{transcript}

Extract the key points from this section:
- Main arguments or claims made
- Stocks, tickers, or assets mentioned and what was said
- Any predictions, data points, or actionable advice
- Any risks or concerns raised

Write 3-5 concise paragraphs."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def _combine_summaries(partials: list, title: str, url: str) -> str:
    combined = "\n\n---\n\n".join(partials)
    prompt = f"""You have partial summaries of different sections of a Meet Kevin video: "{title}"
URL: {url}

PARTIAL SUMMARIES:
{combined}

Write one unified structured summary:

🎯 MAIN THESIS — 1-2 sentences

📌 KEY POINTS — 6-8 points, 2-3 sentences each

📈 STOCKS & TICKERS MENTIONED — every ticker, Kevin's stance and reasoning

💡 ACTIONABLE TAKEAWAYS — 3-5 concrete items

⚠️ RISKS OR CONCERNS MENTIONED — 2-4 points

🔮 FORWARD-LOOKING STATEMENTS — predictions or things to watch

📊 SENTIMENT — one word + one sentence

Be comprehensive and direct. No filler."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ─── Email ────────────────────────────────────────────────────────────────────

def send_email(summary: str, video_title: str, video_url: str, video_published: str, test_mode=False):
    date_str = datetime.now(timezone.utc).strftime("%A, %d %b %Y")
    prefix   = "[TEST] " if test_mode else ""
    subject  = f"{prefix}Meet Kevin: {video_title}"

    html_body = f"""
    <html>
    <body style="font-family: Georgia, serif; max-width: 680px; margin: auto;
                 padding: 24px; color: #1a1a1a; background: #ffffff;">
      {"<div style='background:#fff3cd;padding:10px;border-radius:6px;margin-bottom:16px;font-size:13px;'><b>TEST MODE</b> — This is a test run. seen_videos.json was not updated.</div>" if test_mode else ""}
      <h2 style="color: #1a1a1a; border-bottom: 2px solid #eee; padding-bottom: 12px;">
        📹 Meet Kevin Summary
      </h2>
      <p style="font-size: 18px; font-weight: bold; color: #1a1a2e;">{video_title}</p>
      <p style="font-size: 13px; color: #888;">
        Published: {video_published} &nbsp;|&nbsp;
        <a href="{video_url}" style="color: #4f8ef7;">Watch on YouTube →</a>
      </p>
      <hr style="border: none; border-top: 1px solid #eee; margin: 16px 0;"/>
      <div style="font-size: 15px; line-height: 1.8; white-space: pre-wrap;">{summary}</div>
      <hr style="margin-top: 32px; border: none; border-top: 1px solid #eee;"/>
      <p style="color: #aaa; font-size: 12px; margin-top: 8px;">
        {"TEST RUN — " if test_mode else ""}Auto-summarized via GitHub Actions + Groq (Llama 3.3 70B) • {date_str}
      </p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT
    msg.attach(MIMEText(summary, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

    print(f"  Email sent: {subject}")

# ─── ntfy ─────────────────────────────────────────────────────────────────────

def send_ntfy(video_title: str, success=True, test_mode=False):
    if not KEVIN_NTFY:
        print("  ntfy skipped — KEVIN_NTFY not set.")
        return

    prefix = "[TEST] " if test_mode else ""
    if success:
        title    = f"{prefix}Meet Kevin video summarized"
        message  = f"{video_title} — summary in your inbox."
        tags     = "youtube,white_check_mark"
        priority = "high"
    else:
        title    = "Kevin Bot failed"
        message  = f"Error on: {video_title}. Check GitHub Actions."
        tags     = "warning"
        priority = "urgent"

    try:
        requests.post(
            f"https://ntfy.sh/{KEVIN_NTFY}",
            headers={"Title": title, "Priority": priority, "Tags": tags},
            data=message.encode("utf-8"),
            timeout=10
        )
        print(f"  ntfy sent: {title}")
    except Exception as e:
        print(f"  ntfy warning: {e}")

# ─── Process a single video ───────────────────────────────────────────────────

def process_video(video: dict, test_mode=False):
    print(f"\nProcessing: {video['title']}")
    print(f"  URL: {video['link']}")
    print(f"  Published: {video['published']}")

    print("  Fetching transcript...")
    transcript = get_transcript(video["id"])

    if not transcript:
        print("  No transcript available — skipping.")
        return False

    word_count = len(transcript.split())
    print(f"  Transcript: {word_count:,} words")

    print("  Generating summary with Groq...")
    summary = summarize_transcript(transcript, video["title"], video["link"])

    print("  Sending email...")
    send_email(summary, video["title"], video["link"], video["published"], test_mode=test_mode)

    send_ntfy(video["title"], success=True, test_mode=test_mode)
    return True

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    mode_label = "TEST MODE" if TEST_MODE else "normal run"
    print(f"Kevin Bot starting — {mode_label} — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"Cutoff date: {CUTOFF_DATE.strftime('%B %d %Y')} — older videos will be skipped.")

    print("Fetching Meet Kevin's RSS feed...")
    videos = get_latest_videos(max_videos=5)

    if not videos:
        print("No videos found in RSS feed.")
        return

    print(f"Found {len(videos)} recent videos.")

    # ── TEST MODE ─────────────────────────────────────────────────────────────
    if TEST_MODE:
        print("\nTEST MODE — summarizing the single latest video only.")
        print("seen_videos.json will NOT be updated.")
        latest = videos[0]
        print(f"Latest video: {latest['title']}")
        try:
            process_video(latest, test_mode=True)
            print("\nTest run complete. Check your inbox.")
        except Exception as e:
            print(f"Test run error: {e}")
            send_ntfy(latest["title"], success=False, test_mode=True)
            raise
        return

    # ── NORMAL MODE ───────────────────────────────────────────────────────────
    seen = load_seen_videos()
    print(f"Already processed: {len(seen)} videos")

    new_count = 0
    for video in videos:
        video_id      = video["id"]
        published_dt  = video["published_dt"]

        # Skip if already seen
        if video_id in seen:
            print(f"Skipping (already seen): {video['title']}")
            continue

        # Skip if older than cutoff date
        if published_dt < CUTOFF_DATE:
            print(f"Skipping (before cutoff {CUTOFF_DATE.strftime('%b %d %Y')}): {video['title']} — published {published_dt.strftime('%b %d %Y')}")
            seen.add(video_id)  # Add to seen so we never check it again
            continue

        # New video after cutoff — process it
        try:
            success = process_video(video, test_mode=False)
            seen.add(video_id)
            if success:
                new_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            send_ntfy(video["title"], success=False)
            seen.add(video_id)  # Mark seen to avoid retrying broken videos

    save_seen_videos(seen)
    print(f"\nDone — {new_count} new video(s) summarized. Seen list: {len(seen)} total.")

if __name__ == "__main__":
    main()
