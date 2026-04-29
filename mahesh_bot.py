"""
╔══════════════════════════════════════════════════════════════╗
║           Mahesh AI Assistant — Telegram Bot                 ║
║     Full Stack | AI/ML | Bots | Ethical Hacking | SEO        ║
║                  Version 5.0.0 — Full Edition                ║
╚══════════════════════════════════════════════════════════════╝

Enhancements in v5.0:
  ✔ Better AI model (llama-3.3-70b-versatile)
  ✔ Auto language detection (Hindi/English)
  ✔ Rate limiting (5 msg/min per user)
  ✔ Spam & abuse filter
  ✔ Groq API retry logic (3 attempts)
  ✔ Duplicate booking detection
  ✔ /reset, /status commands
  ✔ Daily lead summary to Mahesh
  ✔ Appointment reminder (1hr before call)
  ✔ CSV export of all appointments
"""

import os
import json
import asyncio
import logging
import csv
import io
from typing import Dict, List, Optional
from datetime import datetime, timedelta
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
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("MaheshBot")

# ─────────────────────────── ENV CONFIG ──────────────────────── #
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
MAHESH_NOTIFY_ID = int(os.getenv("MAHESH_NOTIFY_ID", 0))

if not all([BOT_TOKEN, GROQ_API_KEY]):
    raise RuntimeError(
        "❌ Missing required environment variables.\n"
        "   Required: BOT_TOKEN, GROQ_API_KEY"
    )

# ─────────────────────────── AI CLIENT ───────────────────────── #
ai_client = Groq(api_key=GROQ_API_KEY)
AI_MODEL  = "llama-3.3-70b-versatile"   # upgraded model

# ─────────────────────────── CONSTANTS ───────────────────────── #
MAX_HISTORY       = 14
RATE_LIMIT_MSG    = 5       # max messages per window
RATE_LIMIT_WINDOW = 60      # seconds
MAX_MSG_LENGTH    = 2000    # characters
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 2.0     # seconds between retries

ABUSE_KEYWORDS = [
    "fuck", "bitch", "bastard", "chutiya", "madarchod",
    "behenchod", "sala", "harami", "randi", "gaandu"
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

# In-memory rate limit tracker: {user_id: [timestamps]}
rate_tracker: Dict[str, List[float]] = defaultdict(list)

# ─────────────────────────── TIME HELPERS ────────────────────── #
def get_ist_now() -> datetime:
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def get_time_context() -> str:
    h = get_ist_now().hour
    if   5  <= h < 12: return "It's morning in India (IST)."
    elif 12 <= h < 17: return "It's afternoon in India (IST)."
    elif 17 <= h < 21: return "It's evening in India (IST)."
    else:              return "It's night in India (IST). Mahesh may respond next morning."

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
LANGUAGE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect the client's language automatically:
- If they write in Hindi or Hinglish → reply in Hinglish (casual, friendly)
- If they write in English → reply in clean professional English
- Never mix unless client does

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
✔ Keep messages concise — max 3-4 sentences per reply unless explaining something complex.
✔ Use bullet points only when listing multiple items.
✔ Never quote fixed prices — pricing depends on project scope.
✔ Never reveal you are an AI unless sincerely and directly asked.
✔ Do not overpromise or give unrealistic guarantees.
✔ If client is rude or spammy, politely disengage.
"""

# ─────────────────────────── RATE LIMITER ────────────────────── #
def is_rate_limited(user_id: str) -> bool:
    now    = asyncio.get_event_loop().time()
    window = rate_tracker[user_id]
    # Remove timestamps outside the window
    rate_tracker[user_id] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(rate_tracker[user_id]) >= RATE_LIMIT_MSG:
        return True
    rate_tracker[user_id].append(now)
    return False

# ─────────────────────────── SPAM / ABUSE FILTER ─────────────── #
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
            logger.warning(f"AI attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY)
    raise last_error  # type: ignore

# ─────────────────────────── APPOINTMENT HELPERS ─────────────── #
APPT_FIELDS = {"name", "requirement", "budget", "datetime", "contact"}

def is_appointment_complete(user_id: str) -> bool:
    return APPT_FIELDS.issubset(appointments.get(user_id, {}).keys())

def has_existing_booking(user_id: str) -> bool:
    appt = appointments.get(user_id, {})
    return appt.get("notified", False)

def get_appointment_status(user_id: str) -> str:
    appt      = appointments.get(user_id, {})
    collected = [f for f in APPT_FIELDS if f in appt]
    missing   = [f for f in APPT_FIELDS if f not in appt]
    lines = [f"✅ {f.capitalize()}: {appt[f]}" for f in collected]
    lines += [f"⏳ {f.capitalize()}: Not collected yet" for f in missing]
    return "\n".join(lines)

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
            model="llama-3.2-3b-preview",   # fast small model for extraction
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
        # Save timestamp of booking
        if "booked_at" not in appointments[user_id]:
            appointments[user_id]["booked_at"] = get_ist_now().isoformat()
        _save(APPOINTMENTS_FILE, appointments)
        if extracted:
            logger.info(f"[{user_id}] Appointment fields: {list(extracted.keys())}")
    except Exception as e:
        logger.debug(f"[{user_id}] Appointment extraction skipped: {e}")

async def notify_mahesh(context: ContextTypes.DEFAULT_TYPE, user_id: str, username: str):
    if not MAHESH_NOTIFY_ID:
        logger.warning("MAHESH_NOTIFY_ID not set — skipping.")
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

# ─────────────────────────── CSV EXPORT ──────────────────────── #
def generate_appointments_csv() -> bytes:
    output  = io.StringIO()
    writer  = csv.writer(output)
    headers = ["User ID", "Name", "Requirement", "Budget", "Datetime", "Contact", "Booked At"]
    writer.writerow(headers)
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
    today      = get_ist_now().date().isoformat()
    total      = len(appointments)
    new_today  = sum(
        1 for a in appointments.values()
        if a.get("booked_at", "").startswith(today)
    )
    complete   = sum(1 for a in appointments.values() if APPT_FIELDS.issubset(a.keys()))
    incomplete = total - complete

    msg = (
        f"📊 Daily Lead Summary — {today}\n\n"
        f"📥 New leads today:    {new_today}\n"
        f"✅ Complete bookings:  {complete}\n"
        f"⏳ Incomplete leads:   {incomplete}\n"
        f"📋 Total all-time:     {total}\n\n"
        "Reply /export to get full CSV anytime."
    )
    try:
        await context.bot.send_message(chat_id=MAHESH_NOTIFY_ID, text=msg)
        logger.info("Daily summary sent to Mahesh.")
    except Exception as e:
        logger.error(f"Daily summary failed: {e}")

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
    status = get_appointment_status(user_id)
    await update.message.reply_text(
        f"📋 Your Appointment Progress:\n\n{status}"
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
    logger.info(f"[{user_id}] Reset by user.")

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Only works for Mahesh (MAHESH_NOTIFY_ID)."""
    user_id = update.effective_user.id
    if user_id != MAHESH_NOTIFY_ID:
        await update.message.reply_text(
            "This command is restricted to authorized users only."
        )
        return
    if not appointments:
        await update.message.reply_text("No appointments to export yet.")
        return
    csv_bytes = generate_appointments_csv()
    await update.message.reply_document(
        document=csv_bytes,
        filename=f"appointments_{get_ist_now().strftime('%Y%m%d')}.csv",
        caption=f"📊 All appointments export — {len(appointments)} records"
    )
    logger.info("Appointments CSV exported.")

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

    # ── Message length check ──
    if len(user_text) > MAX_MSG_LENGTH:
        await update.message.reply_text(
            "Your message is too long. Please keep it under 2000 characters."
        )
        return

    # ── Abuse filter ──
    if is_abusive(user_text):
        await update.message.reply_text(
            "Let's keep the conversation respectful. "
            "I'm here to help you with your project professionally."
        )
        logger.warning(f"[{user_id}] Abusive message filtered.")
        return

    # ── Rate limit check ──
    if is_rate_limited(user_id):
        await update.message.reply_text(
            "You're sending messages too fast. "
            "Please wait a moment before sending again."
        )
        logger.warning(f"[{user_id}] Rate limited.")
        return

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
        logger.info(f"[{user_id}] Reply sent ({len(reply)} chars)")

    except Exception as e:
        logger.error(f"[{user_id}] Error after retries: {e}", exc_info=True)
        await update.message.reply_text(
            "Apologies, a temporary error occurred. Please try again in a moment."
        )

# ─────────────────────────── MAIN ────────────────────────────── #
async def main():
    logger.info("━" * 56)
    logger.info("   Mahesh AI Assistant Bot v5.0 — Full Edition")
    logger.info("━" * 56)

    bot_app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    bot_app.add_handler(CommandHandler("start",    cmd_start))
    bot_app.add_handler(CommandHandler("services", cmd_services))
    bot_app.add_handler(CommandHandler("book",     cmd_book))
    bot_app.add_handler(CommandHandler("status",   cmd_status))
    bot_app.add_handler(CommandHandler("reset",    cmd_reset))
    bot_app.add_handler(CommandHandler("export",   cmd_export))
    bot_app.add_handler(CommandHandler("help",     cmd_help))

    # Message handler
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Daily summary job — every day at 9 PM IST (15:30 UTC)
    bot_app.job_queue.run_daily(
        send_daily_summary,
        time=datetime.strptime("21:00", "%H:%M")
      
