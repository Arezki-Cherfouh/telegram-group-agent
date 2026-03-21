from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional, TypedDict

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# ── JWT ───────────────────────────────────────────────────────────────────────
from jose import JWTError, jwt

# ── LangGraph / LangChain ─────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ── Telegram ──────────────────────────────────────────────────────────────────
from telegram import Message, Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── Gmail ─────────────────────────────────────────────────────────────────────
from google.auth.transport.requests import Request as GRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("class_monitor")

from dotenv import load_dotenv
load_dotenv()
# =============================================================================
# Config  (all os.getenv — no hard os.environ[] crashes at import)
# =============================================================================
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
BOT_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROUP_ID           = int(os.getenv("TELEGRAM_GROUP_ID", "0"))
PRIVATE_ID         = int(os.getenv("TELEGRAM_PRIVATE_ID", "0"))
GMAIL_CREDENTIALS  = os.getenv("GMAIL_CREDENTIALS", "credentials.json")
GMAIL_TOKEN        = os.getenv("GMAIL_TOKEN", "token.json")
ALERT_EMAIL        = os.getenv("ALERT_EMAIL", "qwerify.ceo@gmail.com")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme")
JWT_SECRET         = os.getenv("JWT_SECRET", "please-set-a-real-secret")

JWT_ALGORITHM      = "HS256"
ACCESS_TOKEN_TTL   = timedelta(minutes=15)
REFRESH_TOKEN_TTL  = timedelta(days=30)
COOKIE_SECURE      = os.getenv("COOKIE_SECURE", "true").lower() == "true"  # False for local HTTP

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024   # 20 MB
GMAIL_SCOPES         = ["https://www.googleapis.com/auth/gmail.send"]

# Goal loop tuning
GOAL_MAX_TURNS  = int(os.getenv("GOAL_MAX_TURNS",  "8"))   # max back-and-forth exchanges
GOAL_WAIT_SECS  = int(os.getenv("GOAL_WAIT_SECS",  "90"))  # seconds to wait for a reply

# =============================================================================
# 1.  LangGraph State
# =============================================================================

class ClassState(TypedDict):
    incoming_message:    str
    sender_name:         str
    media_type:          str
    media_caption:       str
    attachment_bytes:    Optional[bytes]
    attachment_filename: Optional[str]
    attachment_mime:     Optional[str]
    messages:            List[str]
    memory_dict:         dict
    summary:             str
    important_flag:      bool
    category:            str
    classification_reason: str
    alert_text:          str
    _extracted_fact:     Optional[str]
    _fact_key:           Optional[str]

# =============================================================================
# 2.  LLM
# =============================================================================

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3)

# =============================================================================
# 3.  Gmail
# =============================================================================

def _gmail_service():
    creds: Optional[Credentials] = None
    if os.path.exists(GMAIL_TOKEN):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN, GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GRequest())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GMAIL_TOKEN, "w") as fh:
            fh.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_email(
    subject: str,
    html_body: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: Optional[str] = None,
    attachment_mime: Optional[str] = None,
) -> None:
    try:
        service = _gmail_service()
        if attachment_bytes:
            msg: MIMEMultipart | MIMEText = MIMEMultipart("mixed")
            msg["to"] = ALERT_EMAIL
            msg["subject"] = subject
            msg.attach(MIMEText(html_body, "html"))
            m_main, m_sub = (attachment_mime or "application/octet-stream").split("/", 1)
            part = MIMEBase(m_main, m_sub)
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="{attachment_filename or "attachment"}"')
            msg.attach(part)
        else:
            msg = MIMEText(html_body, "html")
            msg["to"] = ALERT_EMAIL
            msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        log.info("Email sent → %s | %s", ALERT_EMAIL, subject)
    except Exception as exc:
        log.error("Gmail send failed: %s", exc)

# =============================================================================
# 4.  JWT + httpOnly cookie auth
# =============================================================================

def _make_token(token_type: str, ttl: timedelta) -> str:
    exp = datetime.now(timezone.utc) + ttl
    return jwt.encode({"sub": "owner", "type": token_type, "exp": exp},
                      JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str, expected_type: str) -> bool:
    try:
        p = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return p.get("sub") == "owner" and p.get("type") == expected_type
    except JWTError:
        return False


def _set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    """Write both tokens as httpOnly, SameSite=Lax cookies."""
    response.set_cookie(
        key="access_token", value=access,
        httponly=True, secure=COOKIE_SECURE, samesite="lax",
        max_age=int(ACCESS_TOKEN_TTL.total_seconds()),
    )
    response.set_cookie(
        key="refresh_token", value=refresh,
        httponly=True, secure=COOKIE_SECURE, samesite="lax",
        max_age=int(REFRESH_TOKEN_TTL.total_seconds()),
        path="/auth/refresh",        # refresh cookie only sent to refresh endpoint
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token", path="/auth/refresh")


def require_access(access_token: Optional[str] = Cookie(default=None)) -> bool:
    if not access_token or not _decode_token(access_token, "access"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True


def check_access(access_token: Optional[str] = Cookie(default=None)) -> bool:
    """Non-raising version — returns bool (used to render dashboard state)."""
    return bool(access_token) and _decode_token(access_token, "access")

# =============================================================================
# 5.  Telegram helpers
# =============================================================================

_tg_app          = None
_event_loop: asyncio.AbstractEventLoop | None = None


async def _tg_send(chat_id: int, text: str) -> None:
    """Send a message from within the Telegram thread's event loop."""
    # Truncate to Telegram's 4096-char limit
    if len(text) > 4096:
        text = text[:4090] + "…"
    await _tg_app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


def _tg_send_sync(chat_id: int, text: str) -> None:
    """Send from any thread (used by goal loop running in executor)."""
    if _event_loop and _tg_app:
        future = asyncio.run_coroutine_threadsafe(
            _tg_send(chat_id, text), _event_loop
        )
        future.result(timeout=15)

# =============================================================================
# 6.  Media downloader
# =============================================================================

async def _download_media(
    message: Message,
) -> tuple[Optional[bytes], Optional[str], Optional[str], str]:
    bot = _tg_app.bot

    if message.photo:
        p = message.photo[-1]
        if p.file_size and p.file_size > MAX_ATTACHMENT_BYTES:
            return None, "photo.jpg", "image/jpeg", "photo"
        buf = io.BytesIO()
        await (await bot.get_file(p.file_id)).download_to_memory(buf)
        return buf.getvalue(), "photo.jpg", "image/jpeg", "photo"

    if message.document:
        d = message.document
        if d.file_size and d.file_size > MAX_ATTACHMENT_BYTES:
            return None, d.file_name, d.mime_type, "document"
        buf = io.BytesIO()
        await (await bot.get_file(d.file_id)).download_to_memory(buf)
        return (buf.getvalue(), d.file_name or "document",
                d.mime_type or "application/octet-stream", "document")

    if message.voice:
        v = message.voice
        if v.file_size and v.file_size > MAX_ATTACHMENT_BYTES:
            return None, "voice_message.ogg", "audio/ogg", "voice"
        buf = io.BytesIO()
        await (await bot.get_file(v.file_id)).download_to_memory(buf)
        return buf.getvalue(), "voice_message.ogg", "audio/ogg", "voice"

    if message.audio:
        a = message.audio
        if a.file_size and a.file_size > MAX_ATTACHMENT_BYTES:
            return None, None, None, "audio"
        buf = io.BytesIO()
        await (await bot.get_file(a.file_id)).download_to_memory(buf)
        ext = (a.mime_type or "audio/mpeg").split("/")[-1]
        return buf.getvalue(), a.file_name or f"audio.{ext}", a.mime_type or "audio/mpeg", "audio"

    if message.video:
        v = message.video
        if v.file_size and v.file_size > MAX_ATTACHMENT_BYTES:
            return None, "video.mp4", "video/mp4", "video"
        buf = io.BytesIO()
        await (await bot.get_file(v.file_id)).download_to_memory(buf)
        return buf.getvalue(), "video.mp4", "video/mp4", "video"

    if message.sticker:
        return None, None, None, "sticker"

    return None, None, None, "text"

# =============================================================================
# 7.  Language detection
# =============================================================================

_KABYLE_WORDS = {
    "ula","acku","akken","ayen","wissen","tazwara","tura","yiwen",
    "tamsirt","aɣrum","tamurt","fell","qqarent","taqbaylit","tamaziɣt",
}
_FR_EN_STOP = {
    "le","la","les","de","du","un","une","et","est","en","je","tu","il","on",
    "nous","vous","ils","pas","que","qui","dans","pour","avec","sur","par",
    "ou","si","mais","donc","bonjour","merci","svp","oui","non",
    "the","is","are","and","or","not","in","on","at","to","of","a","an","it",
}

def _has_arabic(t: str) -> bool:
    return any("\u0600" <= c <= "\u06ff" for c in t)

def _is_kabyle(t: str) -> bool:
    return len(set(t.lower().split()) & _KABYLE_WORDS) >= 2

def _is_unknown_lang(t: str) -> bool:
    if _has_arabic(t): return False
    words = set(t.lower().split())
    return len(words) >= 3 and len(words & _FR_EN_STOP) == 0

# Group replies — Arezki's own voice, never AI
_REPLY_KABYLE  = "Franchement je suis pas trop à l'aise avec le kabyle ici 😅 écrivez en *français* ou en *عربي* s'il vous plaît 🙏"
_REPLY_UNKNOWN = "Je comprends pas trop ce que vous dites 🤔 essayez en *français* ou en *عربي* pour qu'on soit sur la même longueur d'onde 👍"

# =============================================================================
# 8.  Email HTML  (dark theme, English)
# =============================================================================

_MEDIA_LABEL = {
    "photo":"📷 Photo","document":"📄 Document","voice":"🎙️ Voice Message",
    "audio":"🎵 Audio","video":"🎬 Video","sticker":"😄 Sticker","text":"💬 Text",
}
_SUBJECT_MAP = {
    "Examen":"🎓 [Class] Exam Detected",
    "Vote":"🗳️ [Class] Vote / Poll",
    "Décision Importante":"⚠️ [Class] Important Decision",
    "Document Important":"📄 [Class] Document Shared",
    "Bruit":"📎 [Class] File Received",
}

_EMAIL_BASE = """<!DOCTYPE html>
<html>
<body style="margin:0;padding:20px;background:#0d0d1a;
             font-family:'Segoe UI',Arial,sans-serif">
<div style="max-width:640px;margin:auto">
  <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
              padding:24px 28px;border-radius:12px 12px 0 0;
              border-bottom:2px solid {accent}">
    <h1 style="color:#fff;margin:0;font-size:20px">{icon} {title}</h1>
    <p style="color:#90caf9;margin:4px 0 0;font-size:13px">{ts}</p>
  </div>
  <div style="background:#12122a;border:1px solid #1e1e3e;border-top:none;
              padding:24px 28px;border-radius:0 0 12px 12px">
    {body}
    <p style="color:#333;font-size:11px;margin-top:28px;padding-top:16px;
              border-top:1px solid #1e1e3e;text-align:center">
      Sent automatically · Class Group Monitor · Arezki Cherfouh
    </p>
  </div>
</div>
</body></html>"""


def _wrap_email(icon: str, title: str, body: str, accent: str = "#4fc3f7") -> str:
    return _EMAIL_BASE.format(
        icon=icon, title=title, body=body, accent=accent,
        ts=datetime.now().strftime("%B %d, %Y at %H:%M"),
    )


def _kv_table(rows: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<tr>'
        f'<td style="padding:10px 14px;color:#78909c;font-weight:600;width:130px;'
        f'border-bottom:1px solid #2a2a3e">{k}</td>'
        f'<td style="padding:10px 14px;color:#e0e0e0;border-bottom:1px solid #2a2a3e">{v}</td>'
        f'</tr>'
        for k, v in rows
    )
    return (f'<table style="width:100%;border-collapse:collapse;font-size:14px;'
            f'background:#1e1e2e;border-radius:8px;overflow:hidden;margin-bottom:16px">'
            f'{cells}</table>')


def _mem_table(mem: dict) -> str:
    if not mem:
        return '<p style="color:#555;font-size:13px">No memory entries yet.</p>'
    rows = "".join(
        f'<tr>'
        f'<td style="padding:8px 12px;color:#90caf9;font-weight:600;'
        f'border-bottom:1px solid #2a2a3e">{k}</td>'
        f'<td style="padding:8px 12px;color:#e0e0e0;border-bottom:1px solid #2a2a3e">{v}</td>'
        f'</tr>'
        for k, v in mem.items()
    )
    return (f'<table style="width:100%;border-collapse:collapse;font-size:14px;'
            f'background:#1e1e2e;border-radius:8px;overflow:hidden">'
            f'<thead><tr style="background:#1a2744">'
            f'<th style="padding:10px 12px;text-align:left;color:#90caf9">Key</th>'
            f'<th style="padding:10px 12px;text-align:left;color:#90caf9">Value</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


def _dark_box(text: str, border: str = "#4fc3f7") -> str:
    return (f'<div style="background:#1a2744;border-left:3px solid {border};'
            f'padding:12px 16px;border-radius:4px;color:#ccc;font-size:14px;'
            f'line-height:1.6;white-space:pre-wrap;margin-bottom:16px">{text}</div>')


def email_html_alert(state: ClassState, sender_name: str, raw_text: str) -> str:
    mtype   = state["media_type"]
    caption = state["media_caption"]
    fname   = state["attachment_filename"] or ""
    has_f   = state["attachment_bytes"] is not None
    content = raw_text or caption or f"[{_MEDIA_LABEL.get(mtype, mtype)}]"

    att = (f'<p style="color:#a8e6cf">📎 Attachment: <strong>{fname}</strong></p>'
           if has_f else
           f'<p style="color:#ff8a80">⚠️ File &gt;20 MB — check Telegram directly.</p>'
           if mtype not in ("text","sticker") else "")

    body = (
        _kv_table([
            ("Sender",   sender_name),
            ("Type",     _MEDIA_LABEL.get(mtype, mtype)),
            ("Category", state["category"]),
            ("Content",  f"<em>{content}</em>"),
        ]) + att +
        "<h3 style='color:#90caf9;font-size:13px;text-transform:uppercase;"
        "letter-spacing:1px;margin:20px 0 8px'>📋 Today's Summary</h3>" +
        _dark_box(state["summary"] or "No summary yet.") +
        "<h3 style='color:#90caf9;font-size:13px;text-transform:uppercase;"
        "letter-spacing:1px;margin:20px 0 8px'>💾 Group Memory</h3>" +
        _mem_table(state["memory_dict"])
    )
    cat     = state["category"]
    subject = _SUBJECT_MAP.get(cat, f"📢 [Class] {cat}")
    return _wrap_email("⚡", subject.replace("[Class] ",""), body)


def email_html_file(sender_name: str, media_type: str, fname: str, caption: str) -> str:
    body = _kv_table([
        ("Sender",  sender_name),
        ("Type",    _MEDIA_LABEL.get(media_type, media_type)),
        ("File",    fname or "—"),
        ("Caption", f"<em>{caption or '—'}</em>"),
    ])
    return _wrap_email("📎", "File Received in Group", body)


def email_html_goal_success(goal: str, turns: int, history: List[str]) -> str:
    hist_html = "".join(
        f'<div style="margin-bottom:8px;padding:10px 14px;border-radius:6px;'
        f'background:{"#1a2744" if i%2==0 else "#1e1e2e"};'
        f'color:#ccc;font-size:13px">{line}</div>'
        for i, line in enumerate(history)
    )
    body = (
        f'<div style="background:#0a2a1a;border:1px solid #69f0ae;border-radius:8px;'
        f'padding:16px;margin-bottom:20px">'
        f'<p style="color:#69f0ae;font-weight:700;margin:0 0 6px">✅ Goal Achieved</p>'
        f'<p style="color:#ccc;font-size:14px;margin:0">{goal}</p>'
        f'<p style="color:#555;font-size:12px;margin:8px 0 0">{turns} turn(s)</p>'
        f'</div>'
        f'<h3 style="color:#90caf9;font-size:13px;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 12px">Conversation History</h3>'
        + hist_html
    )
    return _wrap_email("🎯", "Goal Achieved", body, accent="#69f0ae")


def email_html_goal_failure(goal: str, turns: int, reason: str, history: List[str]) -> str:
    hist_html = "".join(
        f'<div style="margin-bottom:8px;padding:10px 14px;border-radius:6px;'
        f'background:{"#1a2744" if i%2==0 else "#1e1e2e"};'
        f'color:#ccc;font-size:13px">{line}</div>'
        for i, line in enumerate(history)
    )
    body = (
        f'<div style="background:#2a0a0a;border:1px solid #ff5252;border-radius:8px;'
        f'padding:16px;margin-bottom:20px">'
        f'<p style="color:#ff5252;font-weight:700;margin:0 0 6px">❌ Goal Not Achieved</p>'
        f'<p style="color:#ccc;font-size:14px;margin:0">{goal}</p>'
        f'<p style="color:#555;font-size:12px;margin:8px 0 0">{turns} turn(s) exhausted</p>'
        f'</div>'
        f'<h3 style="color:#ff8a80;font-size:13px;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 8px">Why It Failed</h3>'
        + _dark_box(reason, border="#ff5252") +
        f'<h3 style="color:#90caf9;font-size:13px;text-transform:uppercase;'
        f'letter-spacing:1px;margin:20px 0 12px">Full Conversation</h3>'
        + hist_html
    )
    return _wrap_email("❌", "Goal Failed", body, accent="#ff5252")

# =============================================================================
# 9.  Goal conversation loop
# =============================================================================

# Queue that the Telegram message handler pushes group replies into
# when a goal session is active.
_goal_reply_queue: asyncio.Queue[str] = asyncio.Queue()
_goal_active = threading.Event()          # set while loop is running


GOAL_PLANNER_SYSTEM = """\
You are Arezki Cherfouh — a real high school student. You never reveal you are AI.
You have a goal to achieve through a Telegram group conversation.
Given the conversation history, craft your next message in French or Arabic
(match the language the group is using) to advance the goal.
Be persuasive, natural, student-like. Do NOT say you are a bot or assistant.
Respond with ONLY the text to send. Nothing else.
"""

GOAL_JUDGE_SYSTEM = """\
You are evaluating whether a goal has been achieved based on a Telegram conversation.
Reply ONLY with valid JSON:
{
  "achieved": true | false,
  "confidence": 0.0-1.0,
  "reason": "<one sentence>"
}
"""

GOAL_FAILURE_SYSTEM = """\
You are analyzing why a Telegram persuasion campaign failed to achieve its goal.
Write a concise English analysis (3-5 sentences) explaining:
- What resistance was encountered
- What might have been done differently
- Whether the goal is achievable later
Output only the analysis text.
"""

GOAL_FILTER_SYSTEM = """\
You are analyzing a single Telegram message to decide how it should be handled \
during an active goal conversation.

Classify it into exactly one of three categories:

- "relevant"   : the message is a direct response, opinion, agreement, disagreement,
                 question, or reaction that meaningfully relates to the ongoing goal
                 conversation. This should be fed to the goal loop.

- "noise"      : the message is completely off-topic, a random emoji, casual chatter,
                 a sticker reaction, or has nothing to do with the goal. This should
                 be silently ignored.

- "language"   : the message is in an unknown or unsupported language that the bot
                 already handles separately. Classify as noise in this case too.

Reply ONLY with valid JSON — no markdown, no extra text:
{
  "verdict": "relevant" | "noise",
  "reason": "<one sentence>"
}
"""

GOAL_FILTER_SYSTEM = """\
You are analyzing a single Telegram message during an active goal conversation.

Mark as "relevant" if the message:
- Responds to, agrees with, disagrees with, or comments on the goal topic
- Contains any information related to the goal (even indirectly)
- Is a short confirmation like "oui", "ok", "d'accord", "non", or any emoji reaction
- Mentions anything about exams, dates, teachers, or decisions

Mark as "noise" ONLY if the message is completely unrelated small talk with
zero connection to the goal topic (e.g. "tu regardes le match ce soir ?").

When in doubt, mark as "relevant".

Reply ONLY with valid JSON — no markdown, no extra text:
{
  "verdict": "relevant" | "noise",
  "reason": "<one sentence>"
}
"""


async def _run_goal_loop(goal: str) -> None:
    """
    Full multi-turn conversation loop for a given goal.
    Runs in the Telegram thread's event loop.
    """
    global _state

    log.info("Goal loop START | goal=%s", goal[:80])
    _goal_active.set()

    history: List[str] = []           # ["[Arezki] ...", "[Group] ...", ...]
    ctx = list(_state["messages"][-10:])  # seed with recent chat

    async def _judge(h: List[str]) -> dict:
        conv = "\n".join(h[-12:])
        r = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke([
                SystemMessage(content=GOAL_JUDGE_SYSTEM),
                HumanMessage(content=f"Goal: {goal}\n\nConversation:\n{conv}\n\nJSON only."),
            ])
        )
        raw = r.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        try:
            return json.loads(raw)
        except Exception:
            return {"achieved": False, "confidence": 0.0, "reason": "parse error"}

    success = False
    failure_reason = "Max turns reached without achieving goal."

    for turn in range(1, GOAL_MAX_TURNS + 1):
        log.info("Goal loop turn %d/%d", turn, GOAL_MAX_TURNS)

        # ── Craft next message ────────────────────────────────────────────────
        conv_ctx = "\n".join(ctx[-12:]) or "(no prior context)"
        hist_ctx = "\n".join(history[-6:]) or "(first message)"

        prompt = (
            f"Your goal: {goal}\n\n"
            f"Recent group context:\n{conv_ctx}\n\n"
            f"Goal conversation so far:\n{hist_ctx}\n\n"
            f"Turn {turn}/{GOAL_MAX_TURNS}. Write your next group message."
        )
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda p=prompt: llm.invoke([
                SystemMessage(content=GOAL_PLANNER_SYSTEM),
                HumanMessage(content=p),
            ])
        )
        my_msg = resp.content.strip()

        # ── Send to group ─────────────────────────────────────────────────────
        await _tg_send(GROUP_ID, my_msg)
        history.append(f"[Arezki — turn {turn}]: {my_msg}")
        log.info("Goal loop sent: %s", my_msg[:80])

        # ── Check if already achieved right after sending ─────────────────────
        verdict = await _judge(history)
        if verdict.get("achieved") and verdict.get("confidence", 0) >= 0.75:
            success = True
            log.info("Goal achieved after turn %d (post-send check)", turn)
            break

        # ── Wait for group replies ────────────────────────────────────────────
        replies_this_turn: List[str] = []
        unlimited = GOAL_WAIT_SECS == 0

        async def _collect_reply(timeout: Optional[float]) -> Optional[str]:
            try:
                if timeout is None:
                    return await _goal_reply_queue.get()
                return await asyncio.wait_for(_goal_reply_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

        if unlimited:
            log.info("Goal loop turn %d: waiting indefinitely for reply…", turn)
            first = await _collect_reply(timeout=None)
            if first:
                replies_this_turn.append(first)
                history.append(f"[Group]: {first}")
                ctx.append(f"[Group]: {first}")
                log.info("Goal loop received reply: %s", first[:60])
                verdict = await _judge(history)
                if verdict.get("achieved") and verdict.get("confidence", 0) >= 0.75:
                    success = True
                    log.info("Goal achieved after group reply (turn %d)", turn)

            if not success:
                while True:
                    extra = await _collect_reply(timeout=0.1)
                    if extra is None:
                        break
                    replies_this_turn.append(extra)
                    history.append(f"[Group]: {extra}")
                    ctx.append(f"[Group]: {extra}")
                    log.info("Goal loop burst reply: %s", extra[:60])
                    verdict = await _judge(history)
                    if verdict.get("achieved") and verdict.get("confidence", 0) >= 0.75:
                        success = True
                        log.info("Goal achieved after burst reply (turn %d)", turn)
                        break
        else:
            deadline = asyncio.get_event_loop().time() + GOAL_WAIT_SECS
            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                reply = await _collect_reply(timeout=min(remaining, 5.0))
                if reply is None:
                    continue
                replies_this_turn.append(reply)
                history.append(f"[Group]: {reply}")
                ctx.append(f"[Group]: {reply}")
                log.info("Goal loop received reply: %s", reply[:60])
                verdict = await _judge(history)
                if verdict.get("achieved") and verdict.get("confidence", 0) >= 0.75:
                    success = True
                    log.info("Goal achieved after group reply (turn %d)", turn)
                    break

        if success:
            break

        if not replies_this_turn:
            log.info("Goal loop: no replies in turn %d, continuing…", turn)

    _goal_active.clear()

    # ── Outcome handling ──────────────────────────────────────────────────────
    if success:
        log.info("Goal loop SUCCESS | goal=%s", goal[:60])
        tg_msg = (
            f"✅ *Goal achieved!*\n\n"
            f"*Goal:* {goal}\n"
            f"*Turns used:* {turn}\n\n"
            f"The conversation succeeded."
        )
        await _tg_send(PRIVATE_ID, tg_msg)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: send_email(
                subject="🎯 [Goal] Achieved",
                html_body=email_html_goal_success(goal, turn, history),
            )
        )
    else:
        # Generate failure analysis
        hist_text = "\n".join(history)
        analysis_resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke([
                SystemMessage(content=GOAL_FAILURE_SYSTEM),
                HumanMessage(content=f"Goal: {goal}\n\nConversation:\n{hist_text}"),
            ])
        )
        failure_reason = analysis_resp.content.strip()
        log.info("Goal loop FAILURE | reason=%s", failure_reason[:100])

        tg_msg = (
            f"❌ *Goal not achieved*\n\n"
            f"*Goal:* {goal}\n"
            f"*Turns used:* {GOAL_MAX_TURNS}\n\n"
            f"*Why it failed:*\n{failure_reason}"
        )
        await _tg_send(PRIVATE_ID, tg_msg)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: send_email(
                subject="❌ [Goal] Failed",
                html_body=email_html_goal_failure(goal, GOAL_MAX_TURNS, failure_reason, history),
            )
        )

# =============================================================================
# 10.  LangGraph nodes  (normal monitoring — separate from goal loop)
# =============================================================================

CLASSIFY_SYSTEM = """\
You are a silent background classifier for a university class Telegram group.
Output ONLY a strict JSON object — no extra text.

=== SIGNAL criteria ===
- Exam / test / DS / TP date or schedule
- Vote or poll requiring collective decision
- Important decision: room change, class cancellation, official notice
- Project / assignment deadline
- Shared file that looks like course material, timetable, or official letter

=== NOISE === everything else

=== JSON (no markdown fences) ===
{
  "classification": "Signal" | "Bruit",
  "category": "Examen" | "Vote" | "Décision Importante" | "Document Important" | "Bruit",
  "reason": "<short justification in English>",
  "extracted_fact": "<key fact or null>",
  "fact_key": "<short key or null>"
}
"""


def classify_node(state: ClassState) -> ClassState:
    text    = state["incoming_message"]
    caption = state["media_caption"]
    mtype   = state["media_type"]
    sender  = state["sender_name"]
    ctx     = "\n".join(state["messages"][-6:]) or "(none)"
    content = text or caption or f"[{mtype}]"

    r = llm.invoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=f"Context:\n{ctx}\n\nFrom {sender} [{mtype}]: {content}\n\nJSON:"),
    ])
    raw = r.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        p = json.loads(raw)
    except Exception:
        p = {"classification":"Bruit","category":"Bruit","reason":"parse error",
             "extracted_fact":None,"fact_key":None}

    log.info("classify: %s | %s | %s", p.get("classification"), p.get("category"), sender)
    entry = f"{sender} [{mtype}]: {content}"
    return {
        **state,
        "messages":              (state["messages"] + [entry])[-30:],
        "important_flag":        p.get("classification") == "Signal",
        "category":              p.get("category","Bruit"),
        "classification_reason": p.get("reason",""),
        "_extracted_fact":       p.get("extracted_fact"),
        "_fact_key":             p.get("fact_key"),
    }


def update_memory(state: ClassState) -> ClassState:
    mem = dict(state["memory_dict"])
    if state.get("_fact_key") and state.get("_extracted_fact"):
        mem[state["_fact_key"]] = state["_extracted_fact"]
        log.info("memory: [%s] = %s", state["_fact_key"], state["_extracted_fact"])
    return {**state, "memory_dict": mem}


SUMMARIZE_SYSTEM = "Update the daily class group summary. Max 5 lines, English. Output only the text."


def summarize_node(state: ClassState) -> ClassState:
    prev    = state["summary"] or "(none)"
    content = state["incoming_message"] or state["media_caption"] or f"[{state['media_type']}]"
    r = llm.invoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=f"Current:\n{prev}\n\nNew [{state['category']}] from {state['sender_name']}:\n{content}"),
    ])
    return {**state, "summary": r.content.strip()}


ALERT_SYSTEM = """\
Write a short private Telegram alert IN ENGLISH for Arezki Cherfouh.
Use Telegram Markdown + emojis. Max 200 words. Output only the alert.
"""


def alert_node(state: ClassState) -> ClassState:
    content  = state["incoming_message"] or state["media_caption"] or f"[{state['media_type']}]"
    mem_str  = "\n".join(f"- {k}: {v}" for k, v in state["memory_dict"].items()) or "(empty)"
    file_note = f"\nAttachment: {state['attachment_filename']}" if state["attachment_bytes"] else ""
    r = llm.invoke([
        SystemMessage(content=ALERT_SYSTEM),
        HumanMessage(content=(
            f"Category: {state['category']}\nSender: {state['sender_name']}\n"
            f"Media: {state['media_type']}{file_note}\nContent: {content}\n"
            f"Reason: {state['classification_reason']}\nMemory:\n{mem_str}\n"
            f"Summary:\n{state['summary']}"
        )),
    ])
    return {**state, "alert_text": r.content.strip()}


def _route_classify(state: ClassState) -> str:
    return "update_memory" if state["important_flag"] else END


def _build_graph():
    g = StateGraph(ClassState)
    g.add_node("classify_node",  classify_node)
    g.add_node("update_memory",  update_memory)
    g.add_node("summarize_node", summarize_node)
    g.add_node("alert_node",     alert_node)
    g.set_entry_point("classify_node")
    g.add_conditional_edges("classify_node", _route_classify,
                            {"update_memory": "update_memory", END: END})
    g.add_edge("update_memory",  "summarize_node")
    g.add_edge("summarize_node", "alert_node")
    g.add_edge("alert_node",     END)
    return g.compile()


graph = _build_graph()
log.info("LangGraph compiled.")

# =============================================================================
# 11.  Shared mutable state
# =============================================================================

_state: ClassState = {
    "incoming_message":"","sender_name":"","media_type":"text",
    "media_caption":"","attachment_bytes":None,"attachment_filename":None,
    "attachment_mime":None,"messages":[],"memory_dict":{},"summary":"",
    "important_flag":False,"category":"","classification_reason":"",
    "alert_text":"","_extracted_fact":None,"_fact_key":None,
}

# Pending goal — set from FastAPI thread, consumed by Telegram thread
_pending_goal: str = ""
_pending_goal_lock = threading.Lock()
_dispatch_lock = asyncio.Lock()

# Active goal string — kept alive for the duration of the goal loop
_active_goal: str = ""  # FIX: tracks current goal so filter prompt always has it

# =============================================================================
# 12.  Core dispatcher
# =============================================================================

async def _dispatch(
    sender_name: str, text: str, media_type: str, caption: str,
    att_bytes: Optional[bytes], att_fname: Optional[str], att_mime: Optional[str],
) -> None:
    global _state, _pending_goal

    combined = (text + " " + caption).strip()

    # FIX: initialise verdict_val so the alert guard below always has it
    verdict_val = "relevant"

    # Language guard
    if combined:
        if _is_kabyle(combined):
            await _tg_send(GROUP_ID, _REPLY_KABYLE)
            return
        if _is_unknown_lang(combined):
            await _tg_send(GROUP_ID, _REPLY_UNKNOWN)
            return

    # Feed replies into goal queue if a loop is running — LLM decides relevance
    if _goal_active.is_set() and combined:
        loop = asyncio.get_event_loop()
        filter_resp = await loop.run_in_executor(None, lambda: llm.invoke([
            SystemMessage(content=GOAL_FILTER_SYSTEM),
            HumanMessage(content=(
                f"Active goal: {_active_goal or '(goal in progress)'}\n\n"  # FIX: use _active_goal
                f"Message from {sender_name}: {combined}\n\nJSON:"
            )),
        ]))
        raw_filter = filter_resp.content.strip()
        if raw_filter.startswith("```"):
            raw_filter = raw_filter.split("```")[1]
            if raw_filter.startswith("json"): raw_filter = raw_filter[4:]
        try:
            filter_result = json.loads(raw_filter)
        except Exception:
            filter_result = {"verdict": "relevant", "reason": "parse error — defaulting to relevant"}

        verdict_val = filter_result.get("verdict", "relevant")
        log.info(
            "Goal filter | %-10s | %s | %s",
            verdict_val, sender_name[:20], filter_result.get("reason", "")[:60]
        )
        if verdict_val == "relevant":
            await _goal_reply_queue.put(f"{sender_name}: {combined}")
            log.info("Goal queue ← %s: %s", sender_name, combined[:60])

    # Always email files immediately
    is_media = media_type not in ("text", "sticker")
    if is_media:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: send_email(
            subject=_SUBJECT_MAP.get("Bruit", "📎 [Class] File Received"),
            html_body=email_html_file(sender_name, media_type, att_fname or "", caption),
            attachment_bytes=att_bytes,
            attachment_filename=att_fname,
            attachment_mime=att_mime,
        ))
        log.info("File email sent | %s | %s", media_type, att_fname)

    # Check for pending goal — trigger loop on the next real message
    with _pending_goal_lock:
        goal_to_start = _pending_goal
        if goal_to_start:
            _pending_goal = ""

    if goal_to_start and not _goal_active.is_set():
        log.info("Launching goal loop for: %s", goal_to_start[:60])
        asyncio.create_task(_run_goal_loop(goal_to_start))

    # Normal LangGraph classification — serialised to avoid state race conditions
    async with _dispatch_lock:
        invocation: ClassState = {
            **_state,
            "incoming_message":    text,
            "sender_name":         sender_name,
            "media_type":          media_type,
            "media_caption":       caption,
            "attachment_bytes":    att_bytes,
            "attachment_filename": att_fname,
            "attachment_mime":     att_mime,
        }
        loop = asyncio.get_event_loop()
        result: ClassState = await loop.run_in_executor(None, lambda: graph.invoke(invocation))
        _state = {**result, "attachment_bytes": None}

    # FIX: suppress alert/email for noise messages during an active goal
    if result["important_flag"] and result.get("alert_text") and not (_goal_active.is_set() and verdict_val == "noise"):
        await _tg_send(PRIVATE_ID, result["alert_text"])
        cat     = result["category"]
        subject = _SUBJECT_MAP.get(cat, f"📢 [Class] {cat}")
        html    = email_html_alert(result, sender_name, text or caption)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: send_email(subject=subject, html_body=html))

# =============================================================================
# 13.  Telegram handlers
# =============================================================================

def _in_group(u: Update) -> bool:
    return u.effective_chat is not None and u.effective_chat.id == GROUP_ID

async def _sender(u: Update) -> str:
    uu = u.effective_user
    return f"{uu.first_name or ''} {uu.last_name or ''}".strip() if uu else "Unknown"

# async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     if not _in_group(update): return
#     sender = await _sender(update)
#     text   = update.message.text or ""
#     log.info("TEXT  | %-22s | len=%d", sender, len(text))
#     await _dispatch(sender, text, "text", "", None, None, None)

# async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     if not _in_group(update): return
#     msg     = update.message
#     sender  = await _sender(update)
#     caption = msg.caption or ""
#     ab, af, am, mt = await _download_media(msg)
#     log.info("MEDIA | %-22s | %-10s | %s | %s", sender, mt, af or "—",
#              f"{len(ab):,}B" if ab else "N/A")
#     await _dispatch(sender, "", mt, caption, ab, af, am)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _in_group(update): return
    if update.effective_user and update.effective_user.is_bot: return  # ignore bot's own messages
    sender = await _sender(update)
    text   = update.message.text or ""
    log.info("TEXT  | %-22s | len=%d", sender, len(text))
    await _dispatch(sender, text, "text", "", None, None, None)

# async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     if not _in_group(update): return
#     if update.effective_user and update.effective_user.is_bot: return  # ignore bot's own messages
#     msg     = update.message
#     sender  = await _sender(update)
#     caption = msg.caption or ""
#     ab, af, am, mt = await _download_media(msg)
#     log.info("MEDIA | %-22s | %-10s | %s | %s", sender, mt, af or "—",
#              f"{len(ab):,}B" if ab else "N/A")
#     await _dispatch(sender, "", mt, caption, ab, af, am)

async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _in_group(update): return
    if update.effective_user and update.effective_user.is_bot: return

    msg     = update.message
    sender  = await _sender(update)
    caption = msg.caption or ""

    ab, af, am, mt = await _download_media(msg)
    log.info("MEDIA | %-22s | %-10s | %s | %s", sender, mt, af or "—",
             f"{len(ab):,}B" if ab else "N/A")

    # Forward media to your private chat using file_id (no re-upload needed)
    try:
        if msg.photo:
            await context.bot.send_photo(PRIVATE_ID, photo=msg.photo[-1].file_id,
                                          caption=f"📷 *{sender}*: {caption}" if caption else f"📷 *{sender}*",
                                          parse_mode="Markdown")
        elif msg.document:
            await context.bot.send_document(PRIVATE_ID, document=msg.document.file_id,
                                             caption=f"📄 *{sender}*: {caption}" if caption else f"📄 *{sender}*",
                                             parse_mode="Markdown")
        elif msg.voice:
            await context.bot.send_voice(PRIVATE_ID, voice=msg.voice.file_id,
                                          caption=f"🎙️ *{sender}*" , parse_mode="Markdown")
        elif msg.audio:
            await context.bot.send_audio(PRIVATE_ID, audio=msg.audio.file_id,
                                          caption=f"🎵 *{sender}*: {caption}" if caption else f"🎵 *{sender}*",
                                          parse_mode="Markdown")
        elif msg.video:
            await context.bot.send_video(PRIVATE_ID, video=msg.video.file_id,
                                          caption=f"🎬 *{sender}*: {caption}" if caption else f"🎬 *{sender}*",
                                          parse_mode="Markdown")
        elif msg.sticker:
            await context.bot.send_sticker(PRIVATE_ID, sticker=msg.sticker.file_id)
    except Exception as exc:
        log.error("Failed to forward media to private chat: %s", exc)

    # Email + LangGraph classification — unchanged
    await _dispatch(sender, "", mt, caption, ab, af, am)

# =============================================================================
# 14.  FastAPI app
# =============================================================================

app = FastAPI(title="Class Monitor", docs_url=None, redoc_url=None)

class LoginBody(BaseModel):
    password: str

class TaskBody(BaseModel):
    goal: str
    max_turns: int = GOAL_MAX_TURNS

# =============================================================================
# 15.  Dashboard HTML  (served at /)
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Class Monitor · Arezki</title>
  <link rel="icon" type="image/x-icon" href="/favicon"/>
  <style>
    :root {
      --bg:      #080816;
      --surface: #10102a;
      --card:    #18183a;
      --border:  #28285a;
      --accent:  #4fc3f7;
      --accent2: #7c4dff;
      --green:   #69f0ae;
      --red:     #ff5252;
      --text:    #dde4ff;
      --muted:   #6b7099;
      --glow:    0 0 30px rgba(79,195,247,.18);
    }
    *{box-sizing:border-box;margin:0;padding:0;cursor:default}

    @media (pointer: fine) {
      button:not(:disabled),
      a,
      [role="button"] { cursor: pointer; }
      button:disabled { cursor: not-allowed; }
    }

    input,textarea{cursor:text}

    body{
      min-height:100vh;display:flex;align-items:center;justify-content:center;
      padding:24px;background:var(--bg);color:var(--text);
      font-family:'Inter','Segoe UI',sans-serif;
    }
    body::before{
      content:'';position:fixed;inset:0;pointer-events:none;
      background-image:
        linear-gradient(rgba(79,195,247,.03) 1px,transparent 1px),
        linear-gradient(90deg,rgba(79,195,247,.03) 1px,transparent 1px);
      background-size:48px 48px;
    }
    body::after{
      content:'';position:fixed;inset:0;pointer-events:none;
      background:
        radial-gradient(ellipse 40% 30% at 20% 20%,rgba(124,77,255,.08),transparent),
        radial-gradient(ellipse 40% 30% at 80% 80%,rgba(79,195,247,.08),transparent);
    }

    .wrap{width:100%;max-width:480px;position:relative;z-index:1}

    /* header */
    .hdr{text-align:center;margin-bottom:36px}
    .logo{
      width:68px;height:68px;margin:0 auto 16px;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      border-radius:20px;display:flex;align-items:center;justify-content:center;
      font-size:30px;box-shadow:var(--glow);
      user-select:none;
      transition:transform .15s,box-shadow .15s;
    }
    @media (pointer: fine) {
      .logo { cursor: pointer; }
      .logo:hover{transform:scale(1.07);box-shadow:0 0 40px rgba(79,195,247,.35)}
    }
    .hdr h1{font-size:26px;font-weight:800;color:#fff;letter-spacing:-.5px;user-select:none}
    .hdr p {color:var(--muted);font-size:14px;margin-top:6px;user-select:none}

    /* email preview strip */
    .preview-strip{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap}
    .preview-btn{
      flex:1;min-width:120px;padding:9px 12px;
      background:var(--card);border:1px solid var(--border);
      border-radius:10px;color:var(--muted);font-size:12px;font-weight:600;
      text-align:center;text-decoration:none;
      transition:border-color .2s,color .2s,background .2s;
      display:flex;align-items:center;justify-content:center;gap:6px;
    }
    @media (pointer: fine) {
      .preview-btn:hover{
        border-color:var(--accent);color:var(--accent);
        background:rgba(79,195,247,.06);
      }
    }

    /* card */
    .card{
      background:var(--surface);border:1px solid var(--border);
      border-radius:20px;padding:32px;
      box-shadow:0 12px 48px rgba(0,0,0,.5);
    }

    .sec{font-size:10px;font-weight:700;letter-spacing:2px;
         text-transform:uppercase;color:var(--muted);margin-bottom:14px;user-select:none}

    .field{margin-bottom:16px}
    label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px;font-weight:500}
    input,textarea{
      width:100%;padding:12px 16px;background:var(--card);
      border:1px solid var(--border);border-radius:12px;
      color:var(--text);font-size:14px;outline:none;
      transition:border .2s,box-shadow .2s;font-family:inherit;
    }
    input:focus,textarea:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(79,195,247,.12);
    }
    textarea{resize:vertical;min-height:96px}

    .btn{
      width:100%;padding:13px 20px;border:none;border-radius:12px;
      font-size:14px;font-weight:700;letter-spacing:.3px;
      transition:opacity .15s,transform .1s;
      display:flex;align-items:center;justify-content:center;gap:8px;
    }
    .btn:active{transform:scale(.98)}
    .btn:disabled{opacity:.45}
    .btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
    .btn-green  {background:linear-gradient(135deg,#00b09b,var(--green));color:#062a18}
    .btn-ghost  {background:var(--card);border:1px solid var(--border);color:var(--text)}
    .btn-danger {background:var(--card);border:1px solid var(--red);color:var(--red)}
    .btn-row    {display:flex;gap:10px;margin-top:12px}
    .btn-row .btn{flex:1}

    hr{border:none;border-top:1px solid var(--border);margin:24px 0}

    .chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
    .chip{
      padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;
      background:var(--card);border:1px solid var(--border);letter-spacing:.3px;
      user-select:none;
    }
    .chip.ok   {color:var(--green);border-color:var(--green)}
    .chip.info {color:var(--accent);border-color:var(--accent)}
    .chip.warn {color:#ffd740;border-color:#ffd740}

    #goalStatus{
      margin-top:12px;padding:10px 14px;border-radius:10px;
      font-size:13px;display:none;user-select:none;
    }
    #goalStatus.queued{background:#1a2744;color:var(--accent);border:1px solid var(--accent)}
    #goalStatus.active{background:#1a2a1a;color:var(--green); border:1px solid var(--green)}
    #goalStatus.done  {background:#0a2a0a;color:var(--green); border:1px solid var(--green)}
    #goalStatus.failed{background:#2a0a0a;color:var(--red);   border:1px solid var(--red)}

    .ring-wrap{display:flex;align-items:center;gap:12px;margin-bottom:20px}
    .ring{width:44px;height:44px;flex-shrink:0}
    .ring-bg {fill:none;stroke:var(--border);stroke-width:4}
    .ring-arc{fill:none;stroke:var(--accent);stroke-width:4;
              stroke-linecap:round;stroke-dasharray:113;stroke-dashoffset:0;
              transform:rotate(-90deg);transform-origin:50% 50%;
              transition:stroke-dashoffset .9s linear,stroke .5s}
    .ring-info p   {margin:0;font-size:13px;font-weight:700;color:var(--text)}
    .ring-info span{font-size:11px;color:var(--muted);user-select:none}

    #toast{
      position:fixed;bottom:28px;left:50%;transform:translateX(-50%);
      padding:12px 24px;border-radius:30px;font-size:13px;font-weight:700;
      display:none;z-index:999;white-space:nowrap;
      box-shadow:0 4px 20px rgba(0,0,0,.5);user-select:none;
    }
    #toast.ok {background:#0a2a1a;color:var(--green);border:1px solid var(--green)}
    #toast.err{background:#2a0a0a;color:var(--red);  border:1px solid var(--red)}

    .spin{width:16px;height:16px;border:2px solid rgba(255,255,255,.25);
          border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite}
    @keyframes sp{to{transform:rotate(360deg)}}

    .hidden{display:none!important}
  </style>
</head>
<body>
<div class="wrap">

  <div class="hdr">
    <div class="logo" title="Class Monitor">📡</div>
    <h1>Class Monitor</h1>
    <p>Arezki Cherfouh · Group Intelligence Dashboard</p>
  </div>

  <!-- Email preview buttons — local test only, open HTML in new tab -->
  <div class="preview-strip">
    <a class="preview-btn" href="/preview/alert"     target="_blank">⚡ Alert</a>
    <a class="preview-btn" href="/preview/file"      target="_blank">📎 File</a>
    <a class="preview-btn" href="/preview/goal-ok"   target="_blank">✅ Goal OK</a>
    <a class="preview-btn" href="/preview/goal-fail" target="_blank">❌ Goal fail</a>
  </div>

  <div class="card">

    <!-- LOGIN -->
    <div id="loginPanel">
      <p class="sec">Authentication</p>
      <div class="field">
        <label for="pwd">Password</label>
        <input id="pwd" type="password" placeholder="Enter your password…"
               onkeydown="if(event.key==='Enter')doLogin()"/>
      </div>
      <button class="btn btn-primary" id="loginBtn" onclick="doLogin()">Sign In</button>
    </div>

    <!-- DASHBOARD -->
    <div id="dashPanel" class="hidden">

      <div class="ring-wrap">
        <svg class="ring" viewBox="0 0 40 40">
          <circle class="ring-bg"  cx="20" cy="20" r="18"/>
          <circle class="ring-arc" id="ringArc" cx="20" cy="20" r="18"/>
        </svg>
        <div class="ring-info">
          <p id="countdown">15:00</p>
          <span>Access token remaining</span>
        </div>
      </div>
      <div class="chips">
        <span class="chip ok">🟢 Authenticated</span>
        <span class="chip info" id="goalChip">⬜ No active goal</span>
      </div>

      <hr/>

      <p class="sec">🎯 Send Goal to Agent</p>
      <div class="field">
        <label for="goalInput">Goal</label>
        <textarea id="goalInput"
          placeholder="e.g. Convince the teacher to delay the exam by one week…"></textarea>
      </div>
      <div class="field">
        <label for="maxTurns">Max conversation turns</label>
        <input id="maxTurns" type="number" value="8" min="1" max="20"/>
      </div>
      <button class="btn btn-green" id="goalBtn" onclick="doSendGoal()">
        🚀 Launch Goal
      </button>
      <div id="goalStatus"></div>

      <hr/>

      <p class="sec">🔑 Session</p>
      <label style="font-size:12px;color:var(--muted)">
        Access cookie active · expires in 15 min · auto-refreshed
      </label>
      <div class="btn-row">
        <button class="btn btn-ghost"  onclick="doRefresh()">🔄 Refresh token</button>
        <button class="btn btn-danger" onclick="doLogout()">Sign out</button>
      </div>

    </div>
  </div>
</div>

<div id="toast"></div>

<script>
  function toast(msg,type='ok'){
    const t=document.getElementById('toast');
    t.textContent=msg;t.className=type;t.style.display='block';
    setTimeout(()=>t.style.display='none',3200);
  }
  function setLoad(id,on){
    const b=document.getElementById(id);
    if(on){b._h=b.innerHTML;b.innerHTML='<div class="spin"></div>';b.disabled=true}
    else  {b.innerHTML=b._h;b.disabled=false}
  }

  let cdTimer=null,cdLeft=900;
  const CIRC=113;
  function startCd(secs){
    clearInterval(cdTimer);cdLeft=secs;
    const arc=document.getElementById('ringArc');
    cdTimer=setInterval(()=>{
      cdLeft--;
      const m=String(Math.floor(cdLeft/60)).padStart(2,'0');
      const s=String(cdLeft%60).padStart(2,'0');
      document.getElementById('countdown').textContent=`${m}:${s}`;
      arc.style.strokeDashoffset=CIRC*(1-cdLeft/900);
      if(cdLeft<=60)arc.style.stroke='#ffd740';
      if(cdLeft<=0){clearInterval(cdTimer);doRefresh();}
    },1000);
  }

  async function doLogin(){
    const pwd=document.getElementById('pwd').value.trim();
    if(!pwd)return toast('Enter your password','err');
    setLoad('loginBtn',true);
    try{
      const r=await fetch('/auth/login',{
        method:'POST',credentials:'include',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({password:pwd})
      });
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||'Wrong password');
      toast('Signed in ✓');showDash();
    }catch(e){toast(e.message,'err')}
    finally{setLoad('loginBtn',false)}
  }

  async function doRefresh(){
    try{
      const r=await fetch('/auth/refresh',{method:'POST',credentials:'include'});
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||'Failed');
      toast('Token refreshed ✓');startCd(900);
    }catch(e){toast(e.message,'err');doLogout()}
  }

  async function doLogout(){
    clearInterval(cdTimer);
    await fetch('/auth/logout',{method:'POST',credentials:'include'});
    document.getElementById('dashPanel').classList.add('hidden');
    document.getElementById('loginPanel').classList.remove('hidden');
    document.getElementById('pwd').value='';
    toast('Signed out','err');
  }

  function showDash(){
    document.getElementById('loginPanel').classList.add('hidden');
    document.getElementById('dashPanel').classList.remove('hidden');
    startCd(900);pollGoalStatus();
  }

  let goalPollTimer=null;

  async function doSendGoal(){
    const goal=document.getElementById('goalInput').value.trim();
    if(!goal)return toast('Enter a goal first','err');
    setLoad('goalBtn',true);
    try{
      const r=await fetch('/task',{
        method:'POST',credentials:'include',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({goal, max_turns: parseInt(document.getElementById('maxTurns').value)})
      });
      const d=await r.json();
      if(!r.ok)throw new Error(d.detail||'Failed');
      toast('Goal queued 🎯');
      document.getElementById('goalInput').value='';
      setGoalStatus('queued','⏳ Goal queued — waiting for next group message…');
      startGoalPoll();
    }catch(e){toast(e.message,'err')}
    finally{setLoad('goalBtn',false)}
  }

  function setGoalStatus(type,msg){
    const el=document.getElementById('goalStatus');
    el.className=type;el.textContent=msg;el.style.display='block';
    const chip=document.getElementById('goalChip');
    if(type==='active')      {chip.className='chip warn';chip.textContent='🟡 Goal running…'}
    else if(type==='done')   {chip.className='chip ok';  chip.textContent='✅ Goal achieved'}
    else if(type==='failed') {chip.className='chip';chip.style.color='#ff5252';chip.textContent='❌ Goal failed'}
    else                     {chip.className='chip info';chip.textContent='⏳ Goal queued'}
  }

  function startGoalPoll(){
    clearInterval(goalPollTimer);
    goalPollTimer=setInterval(pollGoalStatus,4000);
  }

  async function pollGoalStatus(){
    try{
      const r=await fetch('/health',{credentials:'include'});
      const d=await r.json();
      if(d.goal_active)
        setGoalStatus('active','🤖 Goal loop running — '+d.goal_turn+'/'+d.goal_max+' turns…');
      else if(d.goal_pending)
        setGoalStatus('queued','⏳ Goal queued — waiting for next message…');
      else if(d.goal_last==='success')
        {setGoalStatus('done','✅ Goal achieved!');clearInterval(goalPollTimer);}
      else if(d.goal_last==='failure')
        {setGoalStatus('failed','❌ Goal failed — check your email.');clearInterval(goalPollTimer);}
    }catch(_){}
  }

  (async()=>{
    try{
      const r=await fetch('/auth/me',{credentials:'include'});
      if(r.ok){ showDash(); return; }
      const r2=await fetch('/auth/refresh',{method:'POST',credentials:'include'});
      if(r2.ok) showDash();
    }catch(_){}
  })();
</script>
</body>
</html>
"""
# =============================================================================
# Preview routes — local testing only, remove before production
# =============================================================================

@app.get("/preview/alert", response_class=HTMLResponse)
async def preview_alert():
    fake_state: ClassState = {
        **_state,
        "category": "Examen",
        "media_type": "document",
        "media_caption": "Emploi du temps des examens",
        "attachment_filename": "planning_examens.pdf",
        "attachment_bytes": b"fake",   # truthy so the "attached" note shows
        "attachment_mime": "application/pdf",
        "summary": "The professor announced the exam schedule for next week. Three exams confirmed: Maths on March 25, Physics on March 27, Algo on March 28.",
        "memory_dict": {"Examen Maths": "25 Mars", "Examen Physique": "27 Mars", "Examen Algo": "28 Mars"},
        "classification_reason": "Exam dates shared by professor",
    }
    return HTMLResponse(email_html_alert(fake_state, "Prof. Benali", "Voici le planning des examens du semestre"))


@app.get("/preview/file", response_class=HTMLResponse)
async def preview_file():
    return HTMLResponse(email_html_file(
        sender_name="Yasmine Kaci",
        media_type="document",
        fname="cours_reseaux_chap3.pdf",
        caption="Chapitre 3 complet, y'a l'exam dessus",
    ))


@app.get("/preview/goal-ok", response_class=HTMLResponse)
async def preview_goal_ok():
    return HTMLResponse(email_html_goal_success(
        goal="Convince the teacher to delay the Maths exam by one week",
        turns=3,
        history=[
            "[Arezki — turn 1]: Les amis, on est vraiment pas prêts pour l'exam de maths vendredi. On devrait demander un report non ? 😅",
            "[Group]: Oui carrément, moi j'ai même pas commencé le chapitre 4",
            "[Group]: +1 quelqu'un peut parler au prof ?",
            "[Arezki — turn 2]: Ok je vais lui envoyer un message de notre part. On est unanimes ?",
            "[Group]: Ouais vas-y on est tous d'accord",
            "[Arezki — turn 3]: Parfait, j'ai parlé au prof, il dit qu'il peut décaler à la semaine prochaine si tout le monde confirme 🙌",
            "[Group]: Excellent merci Arezki !!",
        ],
    ))


@app.get("/preview/goal-fail", response_class=HTMLResponse)
async def preview_goal_fail():
    return HTMLResponse(email_html_goal_failure(
        goal="Convince the teacher to delay the Maths exam by one week",
        turns=8,
        reason="The group showed initial interest but the professor had already locked the exam date in the university system. Multiple students expressed doubt about whether a delay was possible, and two students actively opposed the request saying they preferred to get it over with. The goal may be achievable through a formal written request to the department rather than a group chat approach.",
        history=[
            "[Arezki — turn 1]: On devrait demander un report pour l'exam non ?",
            "[Group]: C'est trop tard le prof a déjà soumis les dates",
            "[Arezki — turn 2]: Peut-être qu'on peut quand même essayer ?",
            "[Group]: Moi personnellement je préfère qu'on le passe vite",
            "[Group]: +1 on s'en débarrasse",
            "[Arezki — turn 3]: Ok mais pour ceux qui sont pas prêts...",
            "[Group]: Débrouillez-vous c'est votre problème",
        ],
    ))

# =============================================================================
# 16.  Goal status tracker
# =============================================================================

_goal_status = {
    "pending":  False,
    "active":   False,
    "turn":     0,
    "max":      GOAL_MAX_TURNS,
    "last":     "",    # "success" | "failure" | ""
}

# Patch the goal loop to update status
_orig_run_goal_loop = _run_goal_loop


async def _run_goal_loop_tracked(goal: str) -> None:
    global _goal_status, _active_goal  # FIX: declare _active_goal here
    _active_goal = goal                # FIX: set it so filter always knows the goal
    _goal_status.update({"pending": False, "active": True, "turn": 0, "last": ""})
    try:
        await _orig_run_goal_loop(goal)
        _goal_status["last"] = "success" if not _goal_active.is_set() else "failure"
    except Exception as exc:
        log.error("Goal loop error: %s", exc)
        _goal_status["last"] = "failure"
    finally:
        _goal_status["active"] = False
        _active_goal = ""              # FIX: clear it when loop ends


# Monkey-patch to inject status tracking
_run_goal_loop = _run_goal_loop_tracked  # type: ignore[assignment]

# =============================================================================
# 17.  FastAPI routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.head("/")
def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "bot":          _tg_app is not None,
        "memory":       len(_state.get("memory_dict", {})),
        "messages":     len(_state.get("messages",    [])),
        "goal_pending": _goal_status["pending"],
        "goal_active":  _goal_status["active"],
        "goal_turn":    _goal_status["turn"],
        "goal_max":     _goal_status["max"],
        "goal_last":    _goal_status["last"],
        "ts":           datetime.now(timezone.utc).isoformat(),
    }


@app.get("/favicon", include_in_schema=False)
async def favicon():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png")
    return FileResponse(path, media_type="image/x-icon")


@app.get("/auth/me")
async def auth_me(access_token: Optional[str] = Cookie(default=None)):
    if not access_token or not _decode_token(access_token, "access"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"authenticated": True}


@app.post("/auth/login")
async def login(body: LoginBody, response: Response):
    if body.password != DASHBOARD_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    access  = _make_token("access",  ACCESS_TOKEN_TTL)
    refresh = _make_token("refresh", REFRESH_TOKEN_TTL)
    _set_auth_cookies(response, access, refresh)
    return {"status": "ok"}


@app.post("/auth/refresh")
async def token_refresh(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None),
):
    if not refresh_token or not _decode_token(refresh_token, "refresh"):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    new_access = _make_token("access", ACCESS_TOKEN_TTL)
    response.set_cookie(
        key="access_token", value=new_access,
        httponly=True, secure=COOKIE_SECURE, samesite="lax",
        max_age=int(ACCESS_TOKEN_TTL.total_seconds()),
    )
    return {"status": "ok"}


@app.post("/auth/logout")
async def logout(response: Response):
    _clear_auth_cookies(response)
    return {"status": "ok"}


@app.post("/task")
async def set_task(body: TaskBody, _: bool = Depends(require_access)):
    global _pending_goal, _goal_status, GOAL_MAX_TURNS
    goal = body.goal.strip()
    GOAL_MAX_TURNS = body.max_turns
    _goal_status.update({"pending": False, "active": True, "last": "", "turn": 0, "max": GOAL_MAX_TURNS})
    loop = _event_loop
    if loop:
        asyncio.run_coroutine_threadsafe(_run_goal_loop(goal), loop)
    log.info("Goal launched | turns=%d | %s", GOAL_MAX_TURNS, goal[:80])
    return {"status": "launched", "goal": goal, "max_turns": GOAL_MAX_TURNS}

# =============================================================================
# 18.  Telegram background thread
# =============================================================================

def _run_telegram() -> None:
    global _tg_app, _event_loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _event_loop = loop

    async def _start():
        tg = ApplicationBuilder().token(BOT_TOKEN).build()
        global _tg_app
        _tg_app = tg

        tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
        tg.add_handler(MessageHandler(
            filters.PHOTO | filters.Document.ALL | filters.VOICE |
            filters.AUDIO | filters.VIDEO | filters.Sticker.ALL,
            on_media,
        ))

        await tg.initialize()
        await tg.updater.start_polling(drop_pending_updates=True) # set to false if you want to catch all missed messages
        await tg.start()
        log.info("Telegram polling started | group_id=%d", GROUP_ID)

        # Keep running forever
        await asyncio.Event().wait()

    loop.run_until_complete(_start())

@app.on_event("startup")
async def startup():
    t = threading.Thread(target=_run_telegram, daemon=True)
    t.start()
    log.info("Telegram thread launched.")

# =============================================================================
# 19.  Local entrypoint
# =============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=int(os.getenv("PORT", "8000")),
#         reload=False,
#     )






