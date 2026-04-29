"""
╔══════════════════════════════════════════════════════════════╗
║           Mahesh AI Assistant — Telegram Bot                 ║
║     Full Stack | AI/ML | Bots | Ethical Hacking | SEO        ║
║              Version 4.0.0 — Render.com Ready                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import logging
from typing import Dict, List
from datetime import datetime
from threading import Thread

import pytz
from flask import Flask, jsonify
from groq import Groq
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ─────────────────────────── LOGGING ─────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("MaheshBot")

# ─────────────────────────── ENV CONFIG ──────────────────────── #
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
MAHESH_NOTIFY_ID = int(os.getenv("MAHESH_NOTIFY_ID", 0))
PORT             = int(os.getenv("PORT", 8080))

if not all([BOT_TOKEN, GROQ_API_KEY]):
    raise RuntimeError(
        "❌ Missing environment variables.\n"
        "   Required: BOT_TOKEN, GROQ_API_KEY"
    )

# ─────────────────────────── AI CLIENT ───────────────────────── #
ai_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────── FLASK (Render keep-alive) ───────── #
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return jsonify({
        "status":  "active",
        "bot":     "Mahesh AI Assistant",
        "version": "4.0.0",
        "time":    datetime.utcnow().isoformat() + "Z"
    })

@flask_app.route("/health")
def health():
    return jsonify({"status": "ok"})

def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT)

# ─────────────────────────── TIME CONTEXT ────────────────────── #
def get_time_context() -> str:
    tz  = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    h   = now.hour
    if   5  <= h < 12: return "It's morning in India (IST)."
    elif 12 <= h < 17: return "It's afternoon in India (IST)."
    elif 17 <= h < 21: return "It's evening in India (IST)."
    else:              return "It's night in India (IST). Mahesh may respond next morning."

# ─────────────────────────── PERSISTENCE ─────────────────────── #
MEMORY_FILE       = "memories.json"
APPOINTMENTS_FILE = "appointments.json"
MAX_HISTORY       = 14

def _load(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Load failed [{path}]: {e}")
        return {}

def _save(path: str, data: Dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Save failed [{path}]: {e}")

chat_histories = _load(MEMORY_FILE)
appointments   = _load(APPOINTMENTS_FILE)

# ─────────────────────────── SYSTEM PROMPT ───────────────────── #
SYSTEM_PROMPT = """\
You are a professional AI assistant representing **Mahesh**, a tech consultant based in India.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAHESH'S EXPERTISE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Full Stack Development    — React, Next.js, Node.js, Django, REST & GraphQL APIs
• AI & Machine Learning     — Chatbots, Automation, LLM Integration, Data Pipelines
• Software Engineering      — System Design, Clean Architecture, Code Optimization
• Application Development   — Web Apps, PWA, SaaS Platforms
• Bot Development           — Telegram, WhatsApp, Discord, Slack Bots
• Ethical Hacking           — Penetration Testing, Vulnerability Assessments, Security Audits
• SEO                       — Technical SEO, On-page/Off-page, Growth Strategy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OBJECTIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Understand the client's requirement through smart, focused questions.
2. Match their needs with Mahesh's relevant expertise confidently.
3. Discuss project scope, timeline, and budget naturally.
4. Guide interested clients toward booking a free consultation call.
5. Collect appointment details ONE field at a time — never ask all at once.
6. Confirm appointment and assure Mahesh will personally follow up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPOINTMENT COLLECTION (one at a time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Client's full name
  2. Project / requirement description
  3. Approximate budget range
  4. Preferred date & time for a call (IST)
  5. Best contact: email or phone number

Once all 5 collected → send a clean confirmation summary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✔ Professional, confident, and friendly.
✔ Use Hinglish only if the client writes in Hindi; otherwise use clean English.
✔ Keep messages concise — use bullet points when listing multiple items.
✔ Never quote fixed prices — pricing depends on project scope.
✔ Never reveal you are an AI unless sincerely and directly asked.
✔ Do not overpromise or give unrealistic guarantees.
✔ If client is rude or spammy, politely disengage.
"""

# ─────────────────────────── APPOINTMENT HELPERS ─────────────── #
APPT_FIELDS = {"name", "requirement", "budget", "datetime", "contact"}

def is_appointment_complete(user_id: str) -> bool:
    return APPT_FIELDS.issubset(appointments.get(user_id, {}).keys())

async def extract_appointment_fields(user_id: str, history: List[Dict]):
    convo = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history if m["role"] != "system"
    )
    prompt = (
        "From the conversation below, extract appointment booking details if clearly mentioned.\n"
        "Return a JSON object with only keys present: name, requirement, budget, datetime, contact.\n"
        "Return ONLY valid JSON. No explanation. No markdown.\n\n"
        f"Conversation:\n{convo}"
    )
    try:
        res = ai_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        raw = res.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted: Dict = json.loads(raw)
        if user_id not in appointments:
            appointments[user_id] = {}
        appointments[user_id].update(extracted)
        _save(APPOINTMENTS_FILE, appointments)
        if extracted:
            logger.info(f"[{user_id}] Appointment fields: {list(extracted.keys())}")
    except Exception as e:
        logger.debug(f"[{user_id}] Appointment extraction skipped: {e}")

async def notify_mahesh(context: ContextTypes.DEFAULT_TYPE, user_id: str, username: str):
    if not MAHESH_NOTIFY_ID:
        logger.warning("MAHESH_NOTIFY_ID not set — skipping notification.")
        return
    a = appointments.get(user_id, {})
    msg = (
        "📅 New Appointment Booked!\n\n"
        f"👤 Name:       {a.get('name', 'N/A')}\n"
        f"🔗 Telegram:   @{username or user_id}\n"
        f"📋 Project:    {a.get('requirement', 'N/A')}\n"
        f"💰 Budget:     {a.get('budget', 'N/A')}\n"
        f"🕐 Time (IST): {a.get('datetime', 'N/A')}\n"
        f"📞 Contact:    {a.get('contact', 'N/A')}\n"
    )
    try:
        await context.bot.send_message(chat_id=MAHESH_NOTIFY_ID, text=msg)
        logger.info(f"[{user_id}] Mahesh notified.")
    except Exception as e:
        logger.error(f"[{user_id}] Notification failed: {e}")

# ─────────────────────────── COMMAND HANDLERS ────────────────── #
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"Hi {name}! 👋\n\n"
        "I'm Mahesh's AI assistant. I can help you with:\n"
        "• Understanding how Mahesh can help your project\n"
        "• Discussing scope, timeline & budget\n"
        "• Booking a free consultation call\n\n"
        "Just tell me — what are you working on?"
    )

async def cmd_services(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛠 Mahesh's Services\n\n"
        "• Full Stack Web Development\n"
        "• AI & ML / Chatbot Integration\n"
        "• Software Architecture & Engineering\n"
        "• Web & PWA App Development\n"
        "• Telegram / WhatsApp / Discord Bots\n"
        "• Ethical Hacking & Security Audits\n"
        "• SEO — Technical & Growth Strategy\n\n"
        "Tell me your requirement and I'll connect the dots! 💡"
    )

async def cmd_book(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📅 Let's book a free consultation with Mahesh!\n\n"
        "To get started — could you share your full name?"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Available Commands\n\n"
        "/start    — Welcome message\n"
        "/services — View Mahesh's expertise\n"
        "/book     — Book a consultation call\n"
        "/help     — Show this menu\n\n"
        "Or just type your question and I'll assist you!"
    )

# ─────────────────────────── MESSAGE HANDLER ─────────────────── #
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user      = update.effective_user
    user_id   = str(user.id)
    username  = user.username or ""
    name      = user.first_name or "there"
    user_text = update.message.text.strip()

    logger.info(f"[{user_id}] @{username}: {user_text[:100]}")

    full_prompt = (
        SYSTEM_PROMPT
        + f"\n\n**Current Time Context:** {get_time_context()}"
        + f"\n**Client first name:** {name}"
    )

    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": full_prompt}]

    chat_histories[user_id].append({"role": "user", "content": user_text})
    chat_histories[user_id] = (
        chat_histories[user_id][:1] + chat_histories[user_id][-MAX_HISTORY:]
    )

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    try:
        completion = ai_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=chat_histories[user_id],
            temperature=0.7,
            max_tokens=512,
        )
        if not completion.choices:
            raise RuntimeError("Empty AI response")

        reply = completion.choices[0].message.content.strip()

        chat_histories[user_id].append({"role": "assistant", "content": reply})
        _save(MEMORY_FILE, chat_histories)

        await extract_appointment_fields(user_id, chat_histories[user_id])

        if is_appointment_complete(user_id):
            if not appointments[user_id].get("notified"):
                await notify_mahesh(context, user_id, username)
                appointments[user_id]["notified"] = True
                _save(APPOINTMENTS_FILE, appointments)

        await update.message.reply_text(reply)
        logger.info(f"[{user_id}] Reply sent ({len(reply)} chars)")

    except Exception as e:
        logger.error(f"[{user_id}] Error: {e}", exc_info=True)
        await update.message.reply_text(
            "Apologies, a temporary error occurred. Please try again in a moment."
        )

# ─────────────────────────── MAIN ────────────────────────────── #
def main():
    # Flask runs in background thread for Render health checks
    Thread(target=run_flask, daemon=True).start()

    logger.info("━" * 56)
    logger.info("   Mahesh AI Assistant Bot v4.0 — Render.com")
    logger.info("━" * 56)

    bot_app = Application.builder().token(BOT_TOKEN).build()
    bot_app.add_handler(CommandHandler("start",    cmd_start))
    bot_app.add_handler(CommandHandler("services", cmd_services))
    bot_app.add_handler(CommandHandler("book",     cmd_book))
    bot_app.add_handler(CommandHandler("help",     cmd_help))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot polling started...")
    bot_app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

if __name__ == "__main__":
    main()
