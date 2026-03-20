# 📡 Telegram Class Monitor

> Agentic Telegram class group monitor — **LangGraph + Groq (Llama-3 70B) + FastAPI**. Wakes on every message, emails all files instantly (photos, PDFs, voice, video), fires English alerts for exams / votes / deadlines. Ships a dark-theme dashboard with a full multi-turn goal conversation loop. Render-ready, single Python file.

---

## Features

|                                    |                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| 🤖 **Agentic monitoring**          | LangGraph: classify → memory → summarize → alert                                                |
| 📎 **All files emailed instantly** | Photos, PDFs, voice, audio, video — attached before LLM runs                                    |
| ⚡ **Signal detection**            | Strict classifier: exams, votes, decisions, deadlines                                           |
| 🎯 **Multi-turn goal loop**        | POST a goal → agent opens conversation, reads replies, adapts, loops until success or max turns |
| ❌ **Failure analysis**            | If goal not achieved → private Telegram + email with LLM-written failure analysis               |
| 🌙 **Dark-theme emails**           | Styled HTML emails with attachment indicators                                                   |
| 🗣️ **Language guard**              | Group replies as Arezki (never AI) — FR/AR only                                                 |
| 🔐 **httpOnly cookies**            | Access (15 min) + refresh (30 days) — tokens never in JS                                        |
| 🚀 **Render-ready**                | FastAPI + Telegram bot in daemon thread, one file                                               |

---

## Goal Loop Behavior

```
POST /task {"goal": "Convince teacher to delay exam"}
          │
          ▼
 Waits for next group message (trigger)
          │
          ▼
 Turn 1: Agent crafts opening message → sends to group
          │
          ▼
 Waits GOAL_WAIT_SECS for replies → reads them
          │
          ▼
 Judge: goal achieved? ──YES──► ✅ Telegram + email (success)
          │
          NO
          │
          ▼
 Turn 2: Agent reads replies, adapts message → sends
          │  (loops up to GOAL_MAX_TURNS)
          ▼
 Max turns reached ──► ❌ Telegram + email (failure analysis)
```

---

## API Endpoints

| Method | Path            | Auth           | Description                          |
| ------ | --------------- | -------------- | ------------------------------------ |
| `GET`  | `/`             | —              | Dashboard HTML                       |
| `HEAD` | `/`             | —              | Ping (`pong`)                        |
| `GET`  | `/health`       | —              | Status JSON incl. goal state         |
| `GET`  | `/auth/me`      | cookie         | Check session                        |
| `POST` | `/auth/login`   | —              | `{password}` → sets httpOnly cookies |
| `POST` | `/auth/refresh` | refresh cookie | Rotates access token                 |
| `POST` | `/auth/logout`  | —              | Clears cookies                       |
| `POST` | `/task`         | access cookie  | `{goal}` → queue goal                |

---

## Stack

- **[FastAPI](https://fastapi.tiangolo.com)** — REST API + HTML dashboard
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agentic state machine
- **[Groq](https://console.groq.com)** — Llama-3 70B inference
- **[python-telegram-bot](https://python-telegram-bot.org)** — group monitoring
- **[Gmail API](https://developers.google.com/gmail/api)** — file + alert emails
- **[python-jose](https://python-jose.readthedocs.io)** — JWT in httpOnly cookies

---

## Setup

### 1. Install

```bash
pip install fastapi "uvicorn[standard]" langgraph langchain-groq \
            python-telegram-bot google-auth google-auth-oauthlib \
            google-api-python-client "python-jose[cryptography]"
```

### 2. Environment variables

```env
GROQ_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_GROUP_ID=           # negative number
TELEGRAM_PRIVATE_ID=
GMAIL_CREDENTIALS=credentials.json
GMAIL_TOKEN=token.json
ALERT_EMAIL=qwerify.ceo@gmail.com
DASHBOARD_PASSWORD=your_secret_password
JWT_SECRET=some_random_32+_char_string
PORT=8000

# Optional goal tuning
GOAL_MAX_TURNS=8       # max back-and-forth exchanges per goal
GOAL_WAIT_SECS=90      # seconds to wait for group replies per turn
COOKIE_SECURE=true     # set false for local HTTP dev
```

### 3. Gmail OAuth (one-time)

```bash
# Run once locally to generate token.json
python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', ['https://www.googleapis.com/auth/gmail.send'])
creds = flow.run_local_server(port=0)
open('token.json','w').write(creds.to_json())
"
```

Then upload both `credentials.json` and `token.json` as Render secret files.

### 4. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Deploy on Render

1. Push to GitHub
2. New **Web Service** → connect repo
3. **Build:** `pip install -r requirements.txt`
4. **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add all env vars in Render dashboard
6. Upload `credentials.json` + `token.json` as **Secret Files**

### `requirements.txt`

```
fastapi
uvicorn[standard]
langgraph
langchain-groq
python-telegram-bot
google-auth
google-auth-oauthlib
google-api-python-client
python-jose[cryptography]
```

---

## Language policy

| Layer                    | Language                                  |
| ------------------------ | ----------------------------------------- |
| Python logs / internals  | English                                   |
| Bot replies in the group | French / Arabic (as Arezki — never as AI) |
| Private Telegram alerts  | English                                   |
| Emails                   | English                                   |

---
