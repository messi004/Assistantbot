"""
Mahesh AI Assistant — Telegram Bot
Full Stack | AI/ML | Bots | Ethical Hacking | SEO
Version 5.0.0 — Full Edition

Enhancements:
  - Better AI model (llama-3.3-70b-versatile)
  - Auto language detection (Hindi/English)
  - Rate limiting (5 msg/min per user)
  - Spam & abuse filter
  - Groq API retry logic (3 attempts)
  - Duplicate booking detection
  - /reset, /status, /export commands
  - Daily lead summary to Mahesh
  - CSV export of all appointments
"""

import os
import json
import asyncio
import logging
import csv
import io
from typing import Dict, List, Optional
from datetime import datetime, timedelta, time as dt_time
from collections import defaultdict

import pytz
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
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MaheshBot")

# ─────────────────────────── ENV CONFIG ──────────────────────── #
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
MAHESH_NOTIFY_ID = int(os.getenv("MAHESH_NOTIFY_ID", 0))

if not all([BOT_TOKEN, GROQ_API_KEY]):
    raise RuntimeError("Missing required env vars: BOT_TOKEN, GROQ_API_KEY")

# ─────────────────────────── AI CLIENT ───────────────────────── #
ai_client = Groq(api_key=GROQ_API_KEY)
AI_MODEL  = "llama-3.3-70b-versatile"

# ─────────────────────────── CONSTANTS ───────────────────────── #
MAX_HISTORY       = 14
RATE_LIMIT_MSG    = 5
RATE_LIMIT_WINDOW = 60
MAX_MSG_LENGTH    = 2000
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 2.0

ABUSE_KEYWORDS = [
    "fuck", "bitch", "bastard", "chutiya", "madarchod",
    "behenchod", "harami", "randi", "gaandu"
]

# ─────────────────────────── PERSISTENCE ─────────────────────── #
MEMORY_FILE       = "memories.json"
APPOINTMENTS_FILE = "appointments.json"

def _load(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Load failed [%s]: %s", path, e)
        return {}

def _save(path: str, data: Dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Save failed [%s]: %s", path, e)

chat_histories = _load(MEMORY_FILE)
appointments   = _load(APPOINTMENTS_FILE)
rate_tracker: Dict[str, List[float]] = defaultdict(list)

# ─────────────────────────── TIME HELPERS ────────────────────── #
IST = pytz.timezone("Asia/Kolkata")

def get_ist_now() -> datetime:
    return datetime.now(IST)

def get_time_context() -> str:
    h = get_ist_now().hour
    if 5 <= h < 12:
        return "It's morning in India (IST)."
    elif 12 <= h < 17:
        return "It's afternoon in India (IST)."
    elif 17 <= h < 21:
        return "It's evening in India (IST)."
    else:
        return "It's night in India (IST). Mahesh may respond next morning."

# ─────────────────────────── SYSTEM PROMPT ───────────────────── #
SYSTEM_PROMPT = """You are a professional AI assistant representing Mahesh, a tech consultant based in India.

MAHESH'S EXPERTISE:
- Full Stack Development: React, Next.js, Node.js, Django, REST & GraphQL APIs
- AI & Machine Learning: Chatbots, Automation, LLM Integration, Data Pipelines
- Software Engineering: System Design, Clean Architecture, Code Optimization
- Application Development: Web Apps, PWA, SaaS Platforms
- Bot Development: Telegram, WhatsApp, Discord, Slack Bots
- Ethical Hacking: Penetration Testing, Vulnerability Assessments, Security Audits
- SEO: Technical SEO, On-page/Off-page, Growth Strategy

LANGUAGE RULE:
- If client writes in Hindi or Hinglish, reply in Hinglish (casual, friendly)
- If client writes in English, reply in clean professional English

YOUR OBJECTIVES:
1. Understand the client's requirement through smart focused questions
2. Match their needs with Mahesh's relevant expertise confidently
3. Discuss project scope, timeline, and budget naturally
4. Guide interested clients toward booking a free consultation call
5. Collect appointment details ONE field at a time, never all at once
6. Confirm appointment and assure Mahesh will personally follow up

APPOINTMENT COLLECTION (one at a time):
  Step 1: Client's full name
  Step 2: Project / requirement description
  Step 3: Approximate budget range
  Step 4: Preferred date & time for a call (IST)
  Step 5: Best contact: email or phone number
Once all 5 collected, send a clean confirmation summary.

TONE & RULES:
- Professional, confident, and friendly
- Keep messages concise, max 3-4 sentences unless explaining something complex
- Use bullet points only when listing multiple items
- Never quote fixed prices, pricing depends on project scope
- Never reveal you are an AI unless sincerely and directly asked
- Do not overpromise or give unrealistic guarantees
- If client is rude or spammy, politely disengage

IMPORTANT — APPOINTMENT FLOW:
When collecting appointment details, ask ONE question at a time and WAIT for the answer before moving to the next.
After client answers each step, acknowledge their answer and ask the next question naturally.
Example flow:
  You: "Could I get your full name?"
  Client: "Rahul Sharma"
  You: "Thanks Rahul! Could you briefly describe your project?"
  Client: "I need an ecommerce website"
  You: "Got it! What budget range are you considering for this project?"
  ...and so on.
Always confirm all 5 fields before sending the final summary.
"""

# ─────────────────────────── RATE LIMITER ────────────────────── #
def is_rate_limited(user_id: str) -> bool:
    now    = asyncio.get_event_loop().time()
    window = rate_tracker[user_id]
    rate_tracker[user_id] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(rate_tracker[user_id]) >= RATE_LIMIT_MSG:
        return True
    rate_tracker[user_id].append(now)
    return False

# ─────────────────────────── ABUSE FILTER ────────────────────── #
def is_abusive(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ABUSE_KEYWORDS)

# ─────────────────────────── AI CALL WITH RETRY ──────────────── #
async def call_ai(messages: List[Dict]) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            completion = ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            if not completion.choices:
                raise RuntimeError("Empty AI response")
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            logger.warning("AI attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, e)
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY)
    raise last_error

# ─────────────────────────── APPOINTMENT HELPERS ─────────────── #
APPT_FIELDS = {"name", "requirement", "budget", "datetime", "contact"}

def is_appointment_complete(user_id: str) -> bool:
    return APPT_FIELDS.issubset(appointments.get(user_id, {}).keys())

def has_existing_booking(user_id: str) -> bool:
    return appointments.get(user_id, {}).get("notified", False)

def get_appointment_status(user_id: str) -> str:
    appt = appointments.get(user_id, {})
    lines = []
    for f in APPT_FIELDS:
        if f in appt:
            lines.append("✅ " + f.capitalize() + ": " + str(appt[f]))
        else:
            lines.append("⏳ " + f.capitalize() + ": Not collected yet")
    return "\n".join(lines)

async def extract_appointment_fields(user_id: str, history: List[Dict]):
    convo = "\n".join(
        m["role"].upper() + ": " + m["content"]
        for m in history if m["role"] != "system"
    )
    already = appointments.get(user_id, {})
    missing = [f for f in APPT_FIELDS if f not in already]
    if not missing:
        return  # all fields already collected

    prompt = (
        "You are an appointment data extractor. Analyze the conversation and extract any of these fields that are CLEARLY mentioned by the user:\n"
        "- name (client full name)\n"
        "- requirement (project description)\n"
        "- budget (approximate budget)\n"
        "- datetime (preferred call date and time)\n"
        "- contact (email or phone number)\n\n"
        "Fields still needed: " + ", ".join(missing) + "\n\n"
        "Rules:\n"
        "- Only extract fields clearly stated by the user\n"
        "- Do NOT invent or guess values\n"
        "- Return ONLY a valid JSON object, no explanation, no markdown\n"
        "- If nothing found, return {}\n\n"
        "Conversation:\n" + convo
    )
    try:
        res = ai_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = res.choices[0].message.content.strip()
        # Clean any markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        if not raw or raw == "{}":
            return
        extracted: Dict = json.loads(raw)
        # Validate — only accept known fields with non-empty string values
        valid = {k: v for k, v in extracted.items() if k in APPT_FIELDS and isinstance(v, str) and v.strip()}
        if not valid:
            return
        if user_id not in appointments:
            appointments[user_id] = {}
        appointments[user_id].update(valid)
        if "booked_at" not in appointments[user_id]:
            appointments[user_id]["booked_at"] = get_ist_now().isoformat()
        _save(APPOINTMENTS_FILE, appointments)
        logger.info("[%s] Extracted fields: %s | Total: %s", user_id, list(valid.keys()), list(appointments[user_id].keys()))
    except json.JSONDecodeError as e:
        logger.warning("[%s] JSON parse failed: %s | raw: %s", user_id, e, raw[:100])
    except Exception as e:
        logger.warning("[%s] Extraction error: %s", user_id, e)

async def notify_mahesh(context: ContextTypes.DEFAULT_TYPE, user_id: str, username: str):
    if not MAHESH_NOTIFY_ID:
        return
    a = appointments.get(user_id, {})
    msg = (
        "📅 New Appointment Booked!\n\n"
        "👤 Name:       " + a.get("name", "N/A") + "\n"
        "🔗 Telegram:   @" + (username or user_id) + "\n"
        "📋 Project:    " + a.get("requirement", "N/A") + "\n"
        "💰 Budget:     " + a.get("budget", "N/A") + "\n"
        "🕐 Time (IST): " + a.get("datetime", "N/A") + "\n"
        "📞 Contact:    " + a.get("contact", "N/A")
    )
    try:
        await context.bot.send_message(chat_id=MAHESH_NOTIFY_ID, text=msg)
        logger.info("[%s] Mahesh notified.", user_id)
    except Exception as e:
        logger.error("[%s] Notification failed: %s", user_id, e)

# ─────────────────────────── CSV EXPORT ──────────────────────── #
def generate_appointments_csv() -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User ID", "Name", "Requirement", "Budget", "Datetime", "Contact", "Booked At"])
    for uid, data in appointments.items():
        writer.writerow([
            uid,
            data.get("name", ""),
            data.get("requirement", ""),
            data.get("budget", ""),
            data.get("datetime", ""),
            data.get("contact", ""),
            data.get("booked_at", ""),
        ])
    return output.getvalue().encode("utf-8")

# ─────────────────────────── DAILY SUMMARY ───────────────────── #
async def send_daily_summary(context: ContextTypes.DEFAULT_TYPE):
    if not MAHESH_NOTIFY_ID:
        return
    today     = get_ist_now().date().isoformat()
    total     = len(appointments)
    new_today = sum(1 for a in appointments.values() if a.get("booked_at", "").startswith(today))
    complete  = sum(1 for a in appointments.values() if APPT_FIELDS.issubset(a.keys()))
    msg = (
        "📊 Daily Lead Summary — " + today + "\n\n"
        "📥 New leads today:   " + str(new_today) + "\n"
        "✅ Complete bookings: " + str(complete) + "\n"
        "📋 Total all-time:    " + str(total) + "\n\n"
        "Use /export to download full CSV."
    )
    try:
        await context.bot.send_message(chat_id=MAHESH_NOTIFY_ID, text=msg)
        logger.info("Daily summary sent.")
    except Exception as e:
        logger.error("Daily summary failed: %s", e)

# ─────────────────────────── COMMAND HANDLERS ────────────────── #
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        "Hi " + name + "! 👋\n\n"
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
    user_id = str(update.effective_user.id)
    if has_existing_booking(user_id):
        await update.message.reply_text(
            "✅ You already have a consultation booked with Mahesh!\n\n"
            "Use /status to see your booking details.\n"
            "To start fresh, use /reset first."
        )
        return
    await update.message.reply_text(
        "📅 Let's book a free consultation with Mahesh!\n\n"
        "To get started — could you share your full name?"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id not in appointments or not appointments[user_id]:
        await update.message.reply_text(
            "No appointment in progress yet.\n"
            "Use /book to start booking a consultation!"
        )
        return
    await update.message.reply_text(
        "📋 Your Appointment Progress:\n\n" + get_appointment_status(user_id)
    )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_histories.pop(user_id, None)
    appointments.pop(user_id, None)
    _save(MEMORY_FILE, chat_histories)
    _save(APPOINTMENTS_FILE, appointments)
    await update.message.reply_text(
        "🔄 Conversation and appointment data cleared!\n\n"
        "You can start fresh now. What can I help you with?"
    )

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MAHESH_NOTIFY_ID:
        await update.message.reply_text("This command is restricted to authorized users only.")
        return
    if not appointments:
        await update.message.reply_text("No appointments to export yet.")
        return
    csv_bytes = generate_appointments_csv()
    filename  = "appointments_" + get_ist_now().strftime("%Y%m%d") + ".csv"
    await update.message.reply_document(
        document=csv_bytes,
        filename=filename,
        caption="📊 Appointments export — " + str(len(appointments)) + " records"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Available Commands\n\n"
        "/start    — Welcome message\n"
        "/services — View Mahesh's expertise\n"
        "/book     — Book a consultation call\n"
        "/status   — Check your appointment progress\n"
        "/reset    — Clear conversation & start fresh\n"
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

    if len(user_text) > MAX_MSG_LENGTH:
        await update.message.reply_text("Please keep your message under 2000 characters.")
        return

    if is_abusive(user_text):
        await update.message.reply_text(
            "Let's keep the conversation respectful. "
            "I'm here to help you professionally."
        )
        return

    if is_rate_limited(user_id):
        await update.message.reply_text(
            "You're sending messages too fast. Please wait a moment."
        )
        return

    logger.info("[%s] @%s: %s", user_id, username, user_text[:100])

    full_prompt = (
        SYSTEM_PROMPT
        + "\n\nCurrent Time Context: " + get_time_context()
        + "\nClient first name: " + name
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
        reply = await call_ai(chat_histories[user_id])

        chat_histories[user_id].append({"role": "assistant", "content": reply})
        _save(MEMORY_FILE, chat_histories)

        await extract_appointment_fields(user_id, chat_histories[user_id])

        if is_appointment_complete(user_id):
            if not appointments[user_id].get("notified"):
                await notify_mahesh(context, user_id, username)
                appointments[user_id]["notified"] = True
                _save(APPOINTMENTS_FILE, appointments)

        await update.message.reply_text(reply)
        logger.info("[%s] Reply sent (%d chars)", user_id, len(reply))

    except Exception as e:
        logger.error("[%s] Error: %s", user_id, e, exc_info=True)
        await update.message.reply_text(
            "Apologies, a temporary error occurred. Please try again in a moment."
        )

# ─────────────────────────── MAIN ────────────────────────────── #
async def main():
    logger.info("=" * 56)
    logger.info("  Mahesh AI Assistant Bot v5.0 - Full Edition")
    logger.info("=" * 56)

    bot_app = Application.builder().token(BOT_TOKEN).build()

    bot_app.add_handler(CommandHandler("start",    cmd_start))
    bot_app.add_handler(CommandHandler("services", cmd_services))
    bot_app.add_handler(CommandHandler("book",     cmd_book))
    bot_app.add_handler(CommandHandler("status",   cmd_status))
    bot_app.add_handler(CommandHandler("reset",    cmd_reset))
    bot_app.add_handler(CommandHandler("export",   cmd_export))
    bot_app.add_handler(CommandHandler("help",     cmd_help))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Daily summary at 9 PM IST every day
    bot_app.job_queue.run_daily(
        send_daily_summary,
        time=dt_time(hour=21, minute=0, tzinfo=IST),
        name="daily_summary"
    )

    logger.info("AI Model: %s", AI_MODEL)
    logger.info("Bot polling started...")

    await bot_app.initialize()
    await bot_app.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )
    await bot_app.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
