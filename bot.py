import os
import asyncio
import logging
import re
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import FSInputFile, CallbackQuery

from backend import UniversityAgent

logging.basicConfig(level=logging.INFO)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
RAW_DATA_DIR = "./data/raw"

if not TOKEN:
    raise ValueError("Нет TELEGRAM_TOKEN в .env файле")

bot = Bot(token=TOKEN)
dp = Dispatcher()

agent = UniversityAgent()
chat_engine = agent.get_chat_engine()


def is_refusal(text: str) -> bool:
    text_lower = text.lower()
    refusal_phrases = [
        "не найдено", "нет информации", "не удалось найти",
        "not found", "no information", "cannot find", "unable to find",
        "unfortunately", "к сожалению"
    ]
    for phrase in refusal_phrases:
        if phrase in text_lower:
            return True
    return False


def contains_cyrillic(text: str) -> bool:
    return bool(re.search('[а-яА-Я]', text))


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 <b>Hi! I am an AI consultant at Innopolis University.</b>\n\n"
        "I answer questions strictly based on official documents.\n"
        "🇷🇺 Я понимаю русский язык и перевожу ответы.\n"
        "🇬🇧 I speak English."
    )


@dp.message()
async def handle_message(message: types.Message):
    user_query = message.text
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    status_msg = await message.answer("⏳ <i>Analyzing the documents...</i>", parse_mode="HTML")

    try:
        is_russian = contains_cyrillic(user_query)

        final_query = user_query
        if is_russian:
            final_query += " (ВАЖНО: Ответь на этот вопрос СТРОГО НА РУССКОМ ЯЗЫКЕ. Переведи информацию из контекста, если она на английском)."
        else:
            final_query += " (Answer in English)."

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(None, chat_engine.chat, final_query)
        answer_text = str(response)

        if not answer_text.strip() or "Empty Response" in answer_text:
            if is_russian:
                answer_text = "Извините, в документах не найдено информации по вашему запросу."
            else:
                answer_text = "Sorry, no information found in the provided documents."

        builder = InlineKeyboardBuilder()
        seen_files = set()
        has_valid_sources = False

        is_negative = is_refusal(answer_text)

        if response.source_nodes and not is_negative:
            for node in response.source_nodes:
                md_name = node.metadata.get("file_name", "Doc")

                if md_name not in seen_files:
                    pdf_name = md_name.replace(".md", ".pdf")

                    # Обрезка для красоты кнопки
                    short_name = pdf_name if len(pdf_name) < 30 else pdf_name[:27] + "..."

                    safe_callback = f"dl_{pdf_name}"
                    if len(safe_callback.encode('utf-8')) < 64:
                        builder.button(
                            text=f"📥 {short_name}",
                            callback_data=safe_callback
                        )
                        seen_files.add(md_name)
                        has_valid_sources = True

            if has_valid_sources:
                builder.adjust(1)

        if has_valid_sources:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
                text=answer_text,
                reply_markup=builder.as_markup(),
                parse_mode="HTML"
            )
        else:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
                text=answer_text,
                parse_mode="HTML"
            )

    except Exception as e:
        logging.error(f"Error: {e}")
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text=f"❌ Ошибка: {str(e)}"
        )


@dp.callback_query(F.data.startswith("dl_"))
async def download_file(callback: CallbackQuery):
    file_name = callback.data.split("dl_")[1]
    file_path = os.path.join(RAW_DATA_DIR, file_name)

    if not os.path.exists(file_path):
        for f in os.listdir(RAW_DATA_DIR):
            if f.lower() == file_name.lower():
                file_path = os.path.join(RAW_DATA_DIR, f)
                break

    if os.path.exists(file_path):
        try:
            doc = FSInputFile(file_path)
            await callback.message.answer_document(document=doc, caption=f"📄 {file_name}")
            await callback.answer()
        except:
            await callback.answer("Ошибка отправки файла", show_alert=True)
    else:
        await callback.answer("Файл не найден", show_alert=True)


async def main():
    print("🚀 Bot running (Clean UI: Buttons only)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())