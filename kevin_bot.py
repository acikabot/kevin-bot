"""
Meet Kevin YouTube Summarizer Bot
Runs on GitHub Actions — completely free

Every hour it checks Meet Kevin's YouTube channel for new videos.
When a new video is found it:
  1. Fetches the full transcript
  2. Sends it to Groq AI for a deep summary
  3. Emails you the summary
  4. Sends a push notification via ntfy

Tracks already-processed videos in a local JSON file committed back to the repo
so it never summarizes the same video twice.

Free stack:
  - GitHub Actions (scheduler + runner)
  - YouTube RSS feed (no API key needed)
  - youtube-transcript-api (no API key needed)
  - Groq API (free tier — no credit card)
  - Gmail SMTP (free)
  - ntfy.sh (free)
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
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
EMAIL_SENDER    = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD  = os.environ["EMAIL_PASSWORD"]
EMAIL_RECIPIENT = os.environ["EMAIL_RECIPIENT"]
KEVIN_NTFY      = os.environ.get("KEVIN_NTFY", "")

# Meet Kevin's YouTube channel ID
CHANNEL_ID = "UCUvvj5lwue7PspotMDjk5UA"
CHANNEL_RSS = f"https://www.youtube.com/feeds/videos.xml?channel_id={CHANNEL_ID}"

# File that tracks which videos have already been summarized
# This file lives in the repo and gets committed back after each run
SEEN_FILE = "seen_videos.json"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ─── Seen videos tracker ──────────────────────────────────────────────────────

def load_seen_videos() -> set:
    """Load the list of already-processed video IDs from file."""
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("seen", []))
    return set()

def save_seen_videos(seen: set):
    """Save the updated list of processed video IDs back to file."""
    with open(SEEN_FILE, "w") as f:
        json.dump({"seen": list(seen), "last_updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)

# ─── YouTube RSS feed ─────────────────────────────────────────────────────────

def get_latest_videos(max_videos=5):
    """
    Fetch the latest videos from Meet Kevin's channel via RSS.
    Returns a list of dicts with id, title, link, published.
    """
    feed = feedparser.parse(CHANNEL_RSS)
    videos = []
    for entry in feed.entries[:max_videos]:
        # Extract video ID from the entry id field (format: yt:video:VIDEO_ID)
        video_id = entry.get("yt_videoid", "")
        if not video_id:
            # Fallback: parse from the id string
            raw_id = entry.get("id", "")
            video_id = raw_id.split(":")[-1] if ":" in raw_id else ""

        if video_id:
            videos.append({
                "id":        video_id,
                "title":     entry.get("title", "Unknown Title"),
                "link":      f"https://www.youtube.com/watch?v={video_id}",
                "published": entry.get("published", ""),
            })
    return videos

# ─── Transcript fetcher ───────────────────────────────────────────────────────

def get_transcript(video_id: str) -> str:
    """
    Fetch the full transcript for a YouTube video.
    Tries English first, then falls back to auto-generated captions.
    Returns the transcript as a single string, or None if unavailable.
    """
    try:
        # Try to get manually created English transcript first
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            # Fall back to auto-generated
            transcript = transcript_list.find_generated_transcript(["en"])

        # Join all segments into one clean string
        segments = transcript.fetch()
        full_text = " ".join(seg["text"] for seg in segments)

        # Clean up common transcript artifacts
        full_text = re.sub(r"\[.*?\]", "", full_text)  # Remove [Music], [Applause] etc
        full_text = re.sub(r"\s+", " ", full_text).strip()

        return full_text

    except TranscriptsDisabled:
        print(f"Transcripts disabled for video {video_id}")
        return None
    except NoTranscriptFound:
        print(f"No English transcript found for video {video_id}")
        return None
    except Exception as e:
        print(f"Transcript error for {video_id}: {e}")
        return None

# ─── Chunker (for long videos) ────────────────────────────────────────────────

def chunk_transcript(transcript: str, max_chars=12000) -> list:
    """
    Split long transcripts into chunks so they fit within Groq's token limit.
    Each chunk ends at a sentence boundary where possible.
    """
    if len(transcript) <= max_chars:
        return [transcript]

    chunks = []
    while len(transcript) > max_chars:
        # Find the last sentence end within the limit
        cut = transcript[:max_chars].rfind(". ")
        if cut == -1:
            cut = max_chars
        else:
            cut += 1  # Include the period
        chunks.append(transcript[:cut].strip())
        transcript = transcript[cut:].strip()

    if transcript:
        chunks.append(transcript)

    return chunks

# ─── Summarizer ───────────────────────────────────────────────────────────────

def summarize_transcript(transcript: str, video_title: str, video_url: str) -> str:
    """
    Send the transcript to Groq and get a deep structured summary.
    For long transcripts, summarizes in chunks then combines.
    """
    chunks = chunk_transcript(transcript)

    if len(chunks) == 1:
        # Short enough — summarize directly
        return _summarize_chunk(chunks[0], video_title, video_url, is_full=True)
    else:
        # Long video — summarize each chunk then combine into a final summary
        print(f"Long transcript — processing in {len(chunks)} chunks...")
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            partial = _summarize_chunk(chunk, video_title, video_url, is_full=False, chunk_num=i+1, total_chunks=len(chunks))
            chunk_summaries.append(partial)

        # Combine chunk summaries into one final summary
        return _combine_summaries(chunk_summaries, video_title, video_url)

def _summarize_chunk(transcript: str, title: str, url: str, is_full=True, chunk_num=1, total_chunks=1) -> str:
    """Summarize a single chunk of transcript."""

    if is_full:
        prompt = f"""You are summarizing a financial YouTube video by Meet Kevin (Kevin Paffrath), a well-known US financial creator who covers stocks, real estate, the economy, and markets.

VIDEO TITLE: {title}
VIDEO URL: {url}

FULL TRANSCRIPT:
{transcript}

Write a detailed, structured summary with the following sections:

🎯 MAIN THESIS
What is Kevin's central argument or main point in this video? 1-2 sentences.

📌 KEY POINTS
The 5-8 most important points Kevin makes. Each point gets 2-3 sentences of explanation — not just a headline. Include his reasoning and any data or evidence he cites.

📈 STOCKS & TICKERS MENTIONED
List every stock, ETF, or asset mentioned. For each one, note what Kevin said about it — bullish, bearish, neutral, and why. If none mentioned, write "None".

💡 ACTIONABLE TAKEAWAYS
What is Kevin suggesting viewers actually do or watch out for? 3-5 concrete takeaways.

⚠️ RISKS OR CONCERNS MENTIONED
Any risks, warnings, or bearish scenarios Kevin mentioned. 2-4 points.

🔮 FORWARD-LOOKING STATEMENTS
Any predictions, upcoming catalysts, or things Kevin says to watch for in the coming days/weeks.

📊 SENTIMENT
One word — Bullish, Bearish, or Mixed — followed by one sentence explaining the overall tone of the video.

Keep the language direct and informative. No fluff. This summary replaces watching the video."""

    else:
        prompt = f"""This is part {chunk_num} of {total_chunks} of a transcript from a Meet Kevin (financial YouTuber) video titled: "{title}"

TRANSCRIPT SECTION:
{transcript}

Extract and summarize the key points from this section only. Focus on:
- Main arguments or claims made
- Any stocks, tickers, or assets mentioned and what was said about them
- Any predictions, data points, or actionable advice
- Any risks or concerns raised

Write 3-5 concise paragraphs. This will be combined with summaries of other sections later."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def _combine_summaries(chunk_summaries: list, title: str, url: str) -> str:
    """Combine multiple chunk summaries into one final structured summary."""
    combined_text = "\n\n---\n\n".join(chunk_summaries)

    prompt = f"""You have been given partial summaries of different sections of a Meet Kevin YouTube video titled: "{title}"

Here are the partial summaries:
{combined_text}

Now write ONE unified, structured final summary with these sections:

🎯 MAIN THESIS
What is Kevin's central argument? 1-2 sentences.

📌 KEY POINTS
The 6-8 most important points across the whole video. 2-3 sentences each.

📈 STOCKS & TICKERS MENTIONED
Every stock/ETF/asset mentioned across the whole video. What Kevin said about each one.

💡 ACTIONABLE TAKEAWAYS
3-5 concrete things Kevin suggests viewers do or watch.

⚠️ RISKS OR CONCERNS MENTIONED
2-4 risks or warnings Kevin raised.

🔮 FORWARD-LOOKING STATEMENTS
Predictions or upcoming things to watch.

📊 SENTIMENT
One word (Bullish/Bearish/Mixed) + one sentence explanation.

Be comprehensive but direct. No filler."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ─── Email ────────────────────────────────────────────────────────────────────

def send_email(summary: str, video_title: str, video_url: str, video_published: str):
    date_str = datetime.now(timezone.utc).strftime("%A, %d %b %Y")
    subject  = f"📹 Meet Kevin: {video_title}"

    html_body = f"""
    <html>
    <body style="font-family: Georgia, serif; max-width: 680px; margin: auto;
                 padding: 24px; color: #1a1a1a; background: #ffffff;">
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
        Auto-summarized via GitHub Actions + Groq (Llama 3.3 70B) • {date_str}
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

    print(f"Email sent: {subject}")

# ─── ntfy notification ────────────────────────────────────────────────────────

def send_ntfy(video_title: str, video_url: str, success: bool = True):
    if not KEVIN_NTFY:
        print("ntfy skipped — KEVIN_NTFY secret not set.")
        return

    if success:
        title   = "New Meet Kevin video summarized"
        message = f"{video_title} — summary in your inbox."
        tags    = "youtube,white_check_mark"
        priority = "high"
    else:
        title   = "Kevin Bot failed"
        message = f"Error processing: {video_title}. Check GitHub Actions."
        tags    = "warning"
        priority = "urgent"

    try:
        requests.post(
            f"https://ntfy.sh/{KEVIN_NTFY}",
            headers={
                "Title":    title,
                "Priority": priority,
                "Tags":     tags,
            },
            data=message.encode("utf-8"),
            timeout=10
        )
        print(f"ntfy sent: {title}")
    except Exception as e:
        print(f"ntfy warning: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Kevin Bot running at {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    # Load already-seen video IDs
    seen = load_seen_videos()
    print(f"Already processed: {len(seen)} videos")

    # Get latest videos from RSS
    print("Checking Meet Kevin's channel for new videos...")
    videos = get_latest_videos(max_videos=5)

    if not videos:
        print("No videos found in RSS feed — channel may be temporarily unavailable.")
        return

    print(f"Found {len(videos)} recent videos on channel.")

    # Filter to only new ones
    new_videos = [v for v in videos if v["id"] not in seen]
    print(f"New (unseen) videos: {len(new_videos)}")

    if not new_videos:
        print("No new videos since last check. Nothing to do.")
        return

    # Process each new video
    for video in new_videos:
        print(f"\nProcessing: {video['title']}")
        print(f"  URL: {video['link']}")

        try:
            # Fetch transcript
            print("  Fetching transcript...")
            transcript = get_transcript(video["id"])

            if not transcript:
                print("  No transcript available — skipping this video.")
                seen.add(video["id"])  # Mark as seen so we don't retry every hour
                continue

            word_count = len(transcript.split())
            print(f"  Transcript: {word_count:,} words")

            # Summarize
            print("  Generating summary with Groq...")
            summary = summarize_transcript(transcript, video["title"], video["link"])

            # Send email
            print("  Sending email...")
            send_email(summary, video["title"], video["link"], video["published"])

            # Send notification
            send_ntfy(video["title"], video["link"], success=True)

            # Mark as seen
            seen.add(video["id"])
            print(f"  Done: {video['title']}")

        except Exception as e:
            print(f"  Error processing {video['title']}: {e}")
            send_ntfy(video["title"], video["link"], success=False)
            seen.add(video["id"])  # Mark seen to avoid retrying a broken video

    # Save updated seen list
    save_seen_videos(seen)
    print(f"\nSeen list updated — {len(seen)} total videos tracked.")
    print("Done.")

if __name__ == "__main__":
    main()
