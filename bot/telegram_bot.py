"""Low-bandwidth delivery: Telegram bot (free). Voice notes in, voice notes out.
    MV_API=http://localhost:8000 TELEGRAM_TOKEN=... python bot/telegram_bot.py
Users pick a language with /lang yo|ha|ig|en. Sends a text answer first (tiny), then audio.
"""
import os
import tempfile

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

API = os.environ.get("MV_API", "http://localhost:8000")
DISCLAIMER = "General information only. Not a replacement for a health worker."


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Send a text or voice note. Use /lang yo|ha|ig|en.\n{DISCLAIMER}")


async def lang(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    code = (ctx.args or ["en"])[0]
    if code not in {"en", "yo", "ha", "ig"}:
        return await update.message.reply_text("Choose: en, yo, ha, ig")
    ctx.user_data["lang"] = code
    await update.message.reply_text(f"Language set to {code}")


async def text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{API}/ask", json={"text": update.message.text,
                                             "lang": ctx.user_data.get("lang", "en")})
    await update.message.reply_text(r.json()["answer"])


async def voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    f = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
        await f.download_to_drive(tmp.name)
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{API}/ask/voice", data={"lang": ctx.user_data.get("lang", "en")},
                             files={"audio": open(tmp.name, "rb")})
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.write(r.content); out.close()
    await update.message.reply_voice(open(out.name, "rb"))


def main():
    app = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("lang", lang))
    app.add_handler(MessageHandler(filters.VOICE, voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text))
    app.run_polling()


if __name__ == "__main__":
    main()
