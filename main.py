from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional, TypedDict

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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# =============================================================================
# Logging  (English — internal only)
# =============================================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("class_monitor")

# =============================================================================
# Config
# =============================================================================
GROQ_API_KEY      = os.environ["GROQ_API_KEY"]
BOT_TOKEN         = os.environ["TELEGRAM_BOT_TOKEN"]
GROUP_ID          = int(os.environ["TELEGRAM_GROUP_ID"])
PRIVATE_ID        = int(os.environ["TELEGRAM_PRIVATE_ID"])
GMAIL_CREDENTIALS = os.environ.get("GMAIL_CREDENTIALS", "credentials.json")
GMAIL_TOKEN       = os.environ.get("GMAIL_TOKEN", "token.json")
ALERT_EMAIL       = os.environ.get("ALERT_EMAIL", "qwerify.ceo@gmail.com")

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Files larger than this are noted in the email but not attached
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20 MB

# =============================================================================
# 1.  LangGraph State
# =============================================================================

class ClassState(TypedDict):
    # Incoming event
    incoming_message:    str
    sender_name:         str
    # text | photo | document | voice | audio | video | sticker
    media_type:          str
    media_caption:       str
    attachment_bytes:    Optional[bytes]
    attachment_filename: Optional[str]
    attachment_mime:     Optional[str]

    # Rolling window
    messages:    List[str]
    memory_dict: dict
    summary:     str

    # Graph internals
    important_flag:          bool
    category:                str
    classification_reason:   str
    alert_text:              str

    # Carry-fields classify_node → update_memory
    _extracted_fact: Optional[str]
    _fact_key:       Optional[str]

# =============================================================================
# 2.  LLM
# =============================================================================

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

# =============================================================================
# 3.  Gmail
# =============================================================================

def _gmail_service():
    creds: Optional[Credentials] = None
    if os.path.exists(GMAIL_TOKEN):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN, GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_CREDENTIALS, GMAIL_SCOPES
            )
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
    """Send an HTML email with an optional binary attachment."""
    try:
        service = _gmail_service()

        if attachment_bytes:
            msg: MIMEMultipart | MIMEText = MIMEMultipart("mixed")
            msg["to"]      = ALERT_EMAIL
            msg["subject"] = subject
            msg.attach(MIMEText(html_body, "html"))

            m_main, m_sub = (attachment_mime or "application/octet-stream").split("/", 1)
            part = MIMEBase(m_main, m_sub)
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{attachment_filename or "attachment"}"',
            )
            msg.attach(part)
        else:
            msg = MIMEText(html_body, "html")
            msg["to"]      = ALERT_EMAIL
            msg["subject"] = subject

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        log.info("Email sent → %s | subject: %s", ALERT_EMAIL, subject)
    except Exception as exc:
        log.error("Gmail send failed: %s", exc)

# =============================================================================
# 4.  Telegram send helper
# =============================================================================

_tg_app = None  # assigned in main()


async def _tg_send(chat_id: int, text: str) -> None:
    await _tg_app.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="Markdown",
    )

# =============================================================================
# 5.  Media downloader
# =============================================================================

async def _download_media(
    message: Message,
) -> tuple[Optional[bytes], Optional[str], Optional[str], str]:
    """
    Returns (bytes | None, filename | None, mime | None, media_label).
    media_label: photo | document | voice | audio | video | sticker | text
    Returns None bytes when file exceeds MAX_ATTACHMENT_BYTES.
    """
    bot = _tg_app.bot

    if message.photo:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > MAX_ATTACHMENT_BYTES:
            log.warning("Photo too large (%d B) — skipping download", photo.file_size)
            return None, "photo.jpg", "image/jpeg", "photo"
        buf = io.BytesIO()
        await (await bot.get_file(photo.file_id)).download_to_memory(buf)
        return buf.getvalue(), "photo.jpg", "image/jpeg", "photo"

    if message.document:
        doc = message.document
        if doc.file_size and doc.file_size > MAX_ATTACHMENT_BYTES:
            log.warning("Document too large (%d B) — skipping download", doc.file_size)
            return None, doc.file_name, doc.mime_type, "document"
        buf = io.BytesIO()
        await (await bot.get_file(doc.file_id)).download_to_memory(buf)
        return (
            buf.getvalue(),
            doc.file_name or "document",
            doc.mime_type or "application/octet-stream",
            "document",
        )

    if message.voice:
        voice = message.voice
        if voice.file_size and voice.file_size > MAX_ATTACHMENT_BYTES:
            log.warning("Voice too large — skipping")
            return None, "voice_message.ogg", "audio/ogg", "voice"
        buf = io.BytesIO()
        await (await bot.get_file(voice.file_id)).download_to_memory(buf)
        return buf.getvalue(), "voice_message.ogg", "audio/ogg", "voice"

    if message.audio:
        audio = message.audio
        if audio.file_size and audio.file_size > MAX_ATTACHMENT_BYTES:
            log.warning("Audio too large — skipping")
            return None, None, None, "audio"
        buf = io.BytesIO()
        await (await bot.get_file(audio.file_id)).download_to_memory(buf)
        ext   = (audio.mime_type or "audio/mpeg").split("/")[-1]
        fname = audio.file_name or f"audio.{ext}"
        return buf.getvalue(), fname, audio.mime_type or "audio/mpeg", "audio"

    if message.video:
        video = message.video
        if video.file_size and video.file_size > MAX_ATTACHMENT_BYTES:
            log.warning("Video too large (%d B) — skipping download", video.file_size)
            return None, "video.mp4", "video/mp4", "video"
        buf = io.BytesIO()
        await (await bot.get_file(video.file_id)).download_to_memory(buf)
        return buf.getvalue(), "video.mp4", "video/mp4", "video"

    if message.sticker:
        return None, None, None, "sticker"

    return None, None, None, "text"

# =============================================================================
# 6.  Language detection helpers
# =============================================================================

_KABYLE_WORDS = {
    "ula", "acku", "akken", "ayen", "wissen", "tazwara", "tura",
    "yiwen", "tamsirt", "aɣrum", "tamurt", "fell", "qqarent",
    "taqbaylit", "tamaziɣt", "iseɣwan", "agdal",
}

# Arabic unicode block: U+0600–U+06FF
def _has_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in text)

# Basic Latin + common French diacritics only
def _looks_latin_non_french(text: str) -> bool:
    """
    Heuristic: text has Latin chars but no Arabic, no French diacritics,
    and no known French/English stop words — likely an unknown Latin script.
    We use this to trigger the 'please write in FR or AR' reply.
    """
    if _has_arabic(text):
        return False
    fr_en_stop = {
        # French
        "le", "la", "les", "de", "du", "un", "une", "et", "est", "en",
        "je", "tu", "il", "on", "nous", "vous", "ils", "pas", "que",
        "qui", "dans", "pour", "avec", "sur", "par", "ou", "si", "mais",
        "donc", "car", "ni", "or", "bonjour", "merci", "svp", "oui", "non",
        # English
        "the", "is", "are", "was", "and", "or", "not", "in", "on", "at",
        "to", "of", "a", "an", "it", "this", "that", "for", "with",
    }
    words = set(text.lower().split())
    return len(words & fr_en_stop) == 0 and len(words) >= 3

def _is_kabyle(text: str) -> bool:
    words = set(text.lower().split())
    return len(words & _KABYLE_WORDS) >= 2

# Group-facing replies — French/Arabic only (language policy)
_REPLY_KABYLE   = (
    "Je comprends pas trop les messages en kabyle 🙏 "
    "SVP écrivez en *FR* ou en *AR* 😊"
)
_REPLY_UNKNOWN  = (
    "Je comprends pas trop ce message 🤔 "
    "SVP écrivez en *français* ou en *عربي* pour que je puisse vous aider 🙏"
)

# =============================================================================
# 7.  LangGraph nodes
# =============================================================================

CLASSIFY_SYSTEM = """\
You are a silent background classifier for a university class Telegram group.
Analyze the incoming event and output ONLY a strict JSON object — no extra text.

=== SIGNAL criteria (at least one must apply) ===
- Exam / test / DS / TP graded — date or schedule mentioned
- Vote or poll requiring a collective class decision
- Important decision: room change, class cancellation, official professor notice
- Project or assignment deadline
- A shared file (PDF, image, doc) that looks like a course file, exam timetable,
  official letter, or important shared resource

=== Everything else is NOISE ===
Jokes, greetings, short replies ("ok", "merci"), memes, stickers, random voice
notes with no academic content, off-topic chat, emotional reactions.

=== Mandatory JSON (no preamble, no markdown fences) ===
{
  "classification": "Signal" | "Bruit",
  "category": "Examen" | "Vote" | "Décision Importante" | "Document Important" | "Bruit",
  "reason": "<short justification in French>",
  "extracted_fact": "<key fact, or null>",
  "fact_key": "<short memory_dict key, or null>"
}
"""


def classify_node(state: ClassState) -> ClassState:
    text    = state["incoming_message"]
    caption = state["media_caption"]
    mtype   = state["media_type"]
    sender  = state["sender_name"]
    ctx     = "\n".join(state["messages"][-6:]) or "(no history yet)"
    content = text or caption or f"[{mtype} — no text]"

    prompt = (
        f"Recent context (last 6 messages):\n{ctx}\n\n"
        f"---\nNew event from {sender}:\n"
        f"  media_type : {mtype}\n"
        f"  content    : {content}\n\n"
        "Output JSON only."
    )

    response = llm.invoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("classify_node: JSON parse error — defaulting to Bruit. raw=%s", raw[:200])
        parsed = {
            "classification": "Bruit",
            "category": "Bruit",
            "reason": "parse error",
            "extracted_fact": None,
            "fact_key": None,
        }

    is_signal = parsed.get("classification") == "Signal"
    log.info(
        "classify_node: %s | category=%s | sender=%s | reason=%s",
        parsed.get("classification"), parsed.get("category"),
        sender, parsed.get("reason"),
    )

    entry        = f"{sender} [{mtype}]: {content}"
    new_messages = (state["messages"] + [entry])[-30:]

    return {
        **state,
        "messages":              new_messages,
        "important_flag":        is_signal,
        "category":              parsed.get("category", "Bruit"),
        "classification_reason": parsed.get("reason", ""),
        "_extracted_fact":       parsed.get("extracted_fact"),
        "_fact_key":             parsed.get("fact_key"),
    }


def update_memory(state: ClassState) -> ClassState:
    key = state.get("_fact_key")
    val = state.get("_extracted_fact")
    mem = dict(state["memory_dict"])
    if key and val:
        mem[key] = val
        log.info("update_memory: [%s] = %s", key, val)
    return {**state, "memory_dict": mem}


SUMMARIZE_SYSTEM = """\
Update the running daily summary for a university class Telegram group.
Write a concise summary (max 5 lines) IN FRENCH only.
Output ONLY the updated summary text — no preamble, no labels.
"""


def summarize_node(state: ClassState) -> ClassState:
    prev    = state["summary"] or "(Aucun résumé pour l'instant)"
    content = state["incoming_message"] or state["media_caption"] or f"[{state['media_type']}]"

    prompt = (
        f"Résumé actuel :\n{prev}\n\n"
        f"Nouvel événement [{state['category']}] de {state['sender_name']} :\n{content}\n\n"
        "Mets à jour le résumé en français."
    )
    response = llm.invoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    new_summary = response.content.strip()
    log.info("summarize_node: summary updated (%d chars)", len(new_summary))
    return {**state, "summary": new_summary}


ALERT_SYSTEM = """\
You are a personal assistant writing a private Telegram alert for Arezki Cherfouh.
Write a SHORT, well-formatted Telegram Markdown alert IN FRENCH informing him of
an important event in his class group.
Use relevant emojis. Be direct. Max 200 words.
Output ONLY the alert text — no preamble, no subject line.
"""


def alert_node(state: ClassState) -> ClassState:
    content  = state["incoming_message"] or state["media_caption"] or f"[{state['media_type']}]"
    mem_str  = "\n".join(f"- {k}: {v}" for k, v in state["memory_dict"].items()) or "(vide)"
    has_file = state["attachment_bytes"] is not None
    file_note = f"\nFichier joint : {state['attachment_filename']}" if has_file else ""

    prompt = (
        f"Catégorie : {state['category']}\n"
        f"Expéditeur : {state['sender_name']}\n"
        f"Type de média : {state['media_type']}{file_note}\n"
        f"Contenu : {content}\n"
        f"Raison : {state['classification_reason']}\n\n"
        f"Mémoire du groupe :\n{mem_str}\n\n"
        f"Résumé du jour :\n{state['summary']}\n\n"
        "Rédige l'alerte Telegram Markdown."
    )
    response = llm.invoke([
        SystemMessage(content=ALERT_SYSTEM),
        HumanMessage(content=prompt),
    ])
    alert_text = response.content.strip()
    log.info("alert_node: alert generated (%d chars)", len(alert_text))
    return {**state, "alert_text": alert_text}

# =============================================================================
# 8.  Routing
# =============================================================================

def _route_classify(state: ClassState) -> str:
    if state["important_flag"]:
        return "update_memory"
    log.info("routing: noise detected → END (silent)")
    return END

# =============================================================================
# 9.  Build StateGraph
# =============================================================================

def _build_graph():
    g = StateGraph(ClassState)
    g.add_node("classify_node",  classify_node)
    g.add_node("update_memory",  update_memory)
    g.add_node("summarize_node", summarize_node)
    g.add_node("alert_node",     alert_node)

    g.set_entry_point("classify_node")
    g.add_conditional_edges(
        "classify_node",
        _route_classify,
        {"update_memory": "update_memory", END: END},
    )
    g.add_edge("update_memory",  "summarize_node")
    g.add_edge("summarize_node", "alert_node")
    g.add_edge("alert_node",     END)
    return g.compile()


graph = _build_graph()
log.info("LangGraph compiled successfully.")

# =============================================================================
# 10.  Persistent in-memory state
#      Swap messages/memory_dict to shelve/Redis for restart persistence.
# =============================================================================

_state: ClassState = {
    "incoming_message":      "",
    "sender_name":           "",
    "media_type":            "text",
    "media_caption":         "",
    "attachment_bytes":      None,
    "attachment_filename":   None,
    "attachment_mime":       None,
    "messages":              [],
    "memory_dict":           {},
    "summary":               "",
    "important_flag":        False,
    "category":              "",
    "classification_reason": "",
    "alert_text":            "",
    "_extracted_fact":       None,
    "_fact_key":             None,
}

# =============================================================================
# 11.  Email HTML builder  (French — language policy)
# =============================================================================

_SUBJECT_MAP = {
    "Examen":              "🎓 [Classe] Examen détecté",
    "Vote":                "🗳️ [Classe] Vote / Sondage",
    "Décision Importante": "⚠️ [Classe] Décision importante",
    "Document Important":  "📄 [Classe] Document partagé",
    "Bruit":               "📎 [Classe] Fichier reçu",   # files-always path
}

_MEDIA_LABEL = {
    "photo":    "📷 Photo",
    "document": "📄 Document",
    "voice":    "🎙️ Message vocal",
    "audio":    "🎵 Fichier audio",
    "video":    "🎬 Vidéo",
    "sticker":  "😄 Sticker",
    "text":     "💬 Message texte",
}


def _email_html(
    state: ClassState,
    sender_name: str,
    raw_text: str,
    is_signal: bool,
) -> str:
    category   = state["category"]
    mtype      = state["media_type"]
    caption    = state["media_caption"]
    has_bytes  = state["attachment_bytes"] is not None
    fname      = state["attachment_filename"] or ""
    summary    = state["summary"]
    mem_items  = state["memory_dict"]
    is_media   = mtype not in ("text", "sticker")

    content_line = raw_text or caption or f"[{_MEDIA_LABEL.get(mtype, mtype)}]"

    if has_bytes:
        att_note = f'<p>📎 Pièce jointe incluse : <b>{fname}</b></p>'
    elif is_media:
        att_note = (
            '<p style="color:#c0392b">⚠️ Fichier trop volumineux (&gt;20 Mo) — '
            'consulte le groupe Telegram directement.</p>'
        )
    else:
        att_note = ""

    signal_badge = (
        f'<span style="background:#1e3a5f;color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:12px">⚡ {category}</span>'
        if is_signal else
        '<span style="background:#95a5a6;color:#fff;padding:2px 8px;'
        'border-radius:12px;font-size:12px">📎 Fichier (bruit)</span>'
    )

    mem_rows = "".join(
        f'<tr><td style="padding:5px 10px;font-weight:bold;background:#f0f4ff">{k}</td>'
        f'<td style="padding:5px 10px">{v}</td></tr>'
        for k, v in mem_items.items()
    ) or '<tr><td colspan="2" style="padding:5px 10px;color:#999">Aucune donnée</td></tr>'

    now = datetime.now().strftime("%d/%m/%Y à %H:%M")

    mem_section = (
        f"""
        <h3 style="color:#1e3a5f;font-size:15px;margin-top:20px">💾 Mémoire du groupe</h3>
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <thead>
            <tr style="background:#1e3a5f;color:#fff">
              <th style="padding:6px 10px;text-align:left">Clé</th>
              <th style="padding:6px 10px;text-align:left">Valeur</th>
            </tr>
          </thead>
          <tbody>{mem_rows}</tbody>
        </table>
        """
        if is_signal else ""
    )

    summary_section = (
        f"""
        <h3 style="color:#1e3a5f;font-size:15px">📋 Résumé du jour</h3>
        <div style="background:#f8f9ff;padding:12px 16px;border-left:4px solid #1e3a5f;
                    border-radius:4px;font-size:14px;line-height:1.6">
          {summary or "<em style='color:#aaa'>Pas encore de résumé.</em>"}
        </div>
        """
        if is_signal else ""
    )

    return f"""
<html>
<body style="font-family:Arial,sans-serif;color:#1a1a1a;max-width:620px;margin:auto;padding:0">
  <div style="background:#1e3a5f;padding:18px 24px;border-radius:8px 8px 0 0">
    <h2 style="color:#fff;margin:0;font-size:18px">📡 Groupe Classe — Nouveau message</h2>
    <p style="color:#aac4e8;margin:4px 0 0;font-size:13px">{now}</p>
  </div>
  <div style="border:1px solid #dde;border-top:none;padding:20px 24px;border-radius:0 0 8px 8px">
    <p style="margin:0 0 14px">{signal_badge}</p>
    <table style="width:100%;border-collapse:collapse;margin-bottom:14px;font-size:14px">
      <tr>
        <td style="padding:6px 10px;background:#f4f6fb;font-weight:bold;width:140px">Expéditeur</td>
        <td style="padding:6px 10px">{sender_name}</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;background:#f4f6fb;font-weight:bold">Type</td>
        <td style="padding:6px 10px">{_MEDIA_LABEL.get(mtype, mtype)}</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;background:#f4f6fb;font-weight:bold">Contenu</td>
        <td style="padding:6px 10px;font-style:italic">{content_line}</td>
      </tr>
    </table>
    {att_note}
    <hr style="border:none;border-top:1px solid #e0e4ee;margin:16px 0">
    {summary_section}
    {mem_section}
    <p style="color:#bbb;font-size:11px;margin-top:24px;border-top:1px solid #eee;padding-top:10px">
      Envoyé automatiquement · Moniteur Groupe Classe · Arezki Cherfouh
    </p>
  </div>
</body>
</html>
"""

# =============================================================================
# 12.  Core dispatcher  (called for every group message)
# =============================================================================

async def _dispatch(
    sender_name: str,
    text: str,
    media_type: str,
    caption: str,
    att_bytes: Optional[bytes],
    att_fname: Optional[str],
    att_mime:  Optional[str],
) -> None:
    global _state

    combined = (text + " " + caption).strip()

    # ── Language guard — reply in group if message is not understood ──────────
    if combined:
        if _is_kabyle(combined):
            await _tg_send(GROUP_ID, _REPLY_KABYLE)
            log.info("Kabyle detected — replied in group.")
            return
        if _looks_latin_non_french(combined):
            await _tg_send(GROUP_ID, _REPLY_UNKNOWN)
            log.info("Unknown language detected — replied in group.")
            return

    # ── ALWAYS email files regardless of classification ───────────────────────
    has_media = media_type not in ("text", "sticker") and media_type != ""
    if has_media:
        loop = asyncio.get_event_loop()
        file_subject = _SUBJECT_MAP.get("Bruit", "📎 [Classe] Fichier reçu")
        # We'll finalize subject after classification; use a quick placeholder email
        # for the raw file so it arrives immediately, before LLM latency
        quick_html = f"""
<html><body style="font-family:Arial,sans-serif;color:#222;max-width:600px;margin:auto">
  <div style="background:#1e3a5f;padding:14px 20px;border-radius:8px 8px 0 0">
    <h2 style="color:#fff;margin:0;font-size:16px">📎 Fichier reçu dans le groupe</h2>
    <p style="color:#aac4e8;margin:4px 0 0;font-size:12px">
      {datetime.now().strftime('%d/%m/%Y à %H:%M')}
    </p>
  </div>
  <div style="border:1px solid #dde;border-top:none;padding:16px 20px;border-radius:0 0 8px 8px">
    <p><b>Expéditeur :</b> {sender_name}</p>
    <p><b>Type :</b> {_MEDIA_LABEL.get(media_type, media_type)}</p>
    <p><b>Fichier :</b> {att_fname or '—'}</p>
    <p><b>Légende :</b> <em>{caption or '—'}</em></p>
    <p style="color:#aaa;font-size:11px;margin-top:16px">
      Envoyé automatiquement · Moniteur Groupe Classe · Arezki Cherfouh
    </p>
  </div>
</body></html>
"""
        await loop.run_in_executor(
            None,
            lambda: send_email(
                subject=file_subject,
                html_body=quick_html,
                attachment_bytes=att_bytes,
                attachment_filename=att_fname,
                attachment_mime=att_mime,
            ),
        )
        log.info("File email dispatched immediately | type=%s | file=%s", media_type, att_fname)

    # ── Run LangGraph ─────────────────────────────────────────────────────────
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
    result: ClassState = await loop.run_in_executor(
        None, lambda: graph.invoke(invocation)
    )

    # Persist rolling state — drop raw bytes to save memory
    _state = {**result, "attachment_bytes": None}

    # ── If Signal: send smart alert email + private Telegram ─────────────────
    if result["important_flag"] and result.get("alert_text"):
        await _tg_send(PRIVATE_ID, result["alert_text"])
        log.info("Private Telegram alert sent → chat_id=%d", PRIVATE_ID)

        category      = result["category"]
        email_subject = _SUBJECT_MAP.get(category, f"📢 [Classe] {category}")
        html_body     = _email_html(result, sender_name, text or caption, is_signal=True)

        # For signal text messages (no file already sent), attach nothing
        # For signal media, file was already emailed; send a separate context email
        await loop.run_in_executor(
            None,
            lambda: send_email(
                subject=email_subject,
                html_body=html_body,
                # Don't re-attach file — it was already sent in the quick email above
                attachment_bytes=None,
            ),
        )

# =============================================================================
# 13.  Telegram handlers
# =============================================================================

def _in_group(update: Update) -> bool:
    return update.effective_chat is not None and update.effective_chat.id == GROUP_ID


async def _sender(update: Update) -> str:
    u = update.effective_user
    if not u:
        return "Inconnu"
    return f"{u.first_name or ''} {u.last_name or ''}".strip() or "Inconnu"


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _in_group(update):
        return
    sender = await _sender(update)
    text   = update.message.text or ""
    log.info("TEXT  | sender=%-22s | len=%d", sender, len(text))
    await _dispatch(sender, text, "text", "", None, None, None)


async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _in_group(update):
        return
    msg     = update.message
    sender  = await _sender(update)
    caption = msg.caption or ""

    att_bytes, att_fname, att_mime, media_type = await _download_media(msg)
    log.info(
        "MEDIA | sender=%-22s | type=%-10s | file=%-28s | size=%s",
        sender, media_type, att_fname or "—",
        f"{len(att_bytes):,} B" if att_bytes else "N/A (too large)",
    )

    await _dispatch(sender, "", media_type, caption, att_bytes, att_fname, att_mime)

# =============================================================================
# 14.  Entry point
# =============================================================================

def main() -> None:
    global _tg_app

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    _tg_app = app

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(
        (
            filters.PHOTO
            | filters.Document.ALL
            | filters.VOICE
            | filters.AUDIO
            | filters.VIDEO
            | filters.Sticker.ALL
        ),
        on_media,
    ))

    log.info("Bot started. Monitoring group_id=%d", GROUP_ID)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()