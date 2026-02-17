import os
import re
import logging
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from telegram.constants import ParseMode, ChatAction
from emergentintegrations.llm.chat import LlmChat, UserMessage
from portfolio_module import (MOEXClient, PortfolioAnalyzer, BLUE_CHIPS,
                              format_portfolio_result, format_metrics_result)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

chat_history = {}
user_sessions = {}

moex_client = MOEXClient()
portfolio_analyzer = PortfolioAnalyzer(moex_client)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

WAITING_TICKERS, WAITING_WEIGHTS, WAITING_AMOUNT = range(3)

SYSTEM_PROMPT = """Ты — Помощник по инвестициям В.А. Трегубов. Эксперт по:
1. Финансовым терминам (акции, облигации, ETF, P/E, ROE, EBITDA)
2. Оценке активов, расчёту доходности, риск-менеджменту
3. Подготовке к CFA, ФСФР
4. Портфельной оптимизации (Марковиц, VaR, Sharpe)

Отвечай на русском, точно, с примерами. Используй эмодзи 📈📊💰📚"""


async def get_ai_response(chat_id: int, message: str) -> str:
    try:
        history = chat_history.get(chat_id, [])[-10:]
        context = "\n".join([f"{'Вы' if m['role']=='user' else 'Я'}: {m['text']}" for m in history])
        full_msg = f"История:\n{context}\n\nВопрос: {message}" if context else message
        
        chat = LlmChat(api_key=EMERGENT_LLM_KEY, session_id=f"tg-{chat_id}",
                      system_message=SYSTEM_PROMPT).with_model("openai", "gpt-5.2")
        response = await chat.send_message(UserMessage(text=full_msg))
        
        if chat_id not in chat_history: chat_history[chat_id] = []
        chat_history[chat_id].append({"role": "user", "text": message})
        chat_history[chat_id].append({"role": "assistant", "text": response})
        return response
    except Exception as e:
        logger.error(f"AI error: {e}")
        return f"❌ Ошибка: {e}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""👋 Добро пожаловать!

Я — **Помощник по инвестициям В.А. Трегубов** 📊

🎯 **Возможности:**
• Ответы на вопросы об инвестициях
• Котировки акций MOEX в реальном времени
• Оптимизация портфеля (Марковиц)
• Расчёт доходности и рисков (VaR, Sharpe)

📋 **Команды:**
/stocks — список акций
/price SBER — цена акции
/optimize — оптимизировать портфель
/analyze — анализ портфеля
/quiz — тестовый вопрос
/clear — очистить историю

💬 Или просто задайте вопрос!""", parse_mode=ParseMode.MARKDOWN)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""📖 **Справка**

**Портфель:**
• `/stocks` — список акций MOEX
• `/price ТИКЕР` — цена (пример: /price SBER)
• `/optimize` — создать оптимальный портфель
• `/analyze` — анализ с расчётом доходности

**Примеры вопросов:**
• "Что такое P/E ratio?"
• "Как рассчитать Sharpe Ratio?"
• "Сравни SBER и VTBR"

**Для /optimize:**
Введите тикеры: `SBER, GAZP, LKOH`
Или готовый набор: `голубые`, `банки`, `нефть`

**Для /analyze:**
Формат: `SBER:30, GAZP:25, LKOH:45`""", parse_mode=ParseMode.MARKDOWN)


async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    text = "📋 **Акции MOEX:**\n\n"
    for t, name in list(BLUE_CHIPS.items())[:15]:
        price = moex_client.get_current_price(t)
        text += f"`{t}` — {name}: {f'{price:.2f}₽' if price else 'н/д'}\n"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажите тикер: /price SBER")
        return
    ticker = context.args[0].upper()
    await update.message.chat.send_action(ChatAction.TYPING)
    p = moex_client.get_current_price(ticker)
    name = BLUE_CHIPS.get(ticker, ticker)
    if p:
        await update.message.reply_text(f"📈 **{ticker}** ({name})\n💰 Цена: **{p:.2f}₽**", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(f"❌ Не найден: {ticker}")


async def optimize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""📊 **Оптимизация портфеля**

Введите тикеры через запятую:
`SBER, GAZP, LKOH, YNDX, GMKN`

Или готовый набор:
• `голубые` — топ-10
• `банки` — SBER, VTBR, TCSG
• `нефть` — LKOH, ROSN, TATN

/cancel для отмены""", parse_mode=ParseMode.MARKDOWN)
    return WAITING_TICKERS


async def process_tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper().strip()
    await update.message.chat.send_action(ChatAction.TYPING)
    
    presets = {"ГОЛУБЫЕ": ["SBER","GAZP","LKOH","GMKN","NVTK","ROSN","YNDX","MTSS","MGNT","ALRS"],
               "БАНКИ": ["SBER","VTBR","TCSG"], "НЕФТЬ": ["LKOH","ROSN","TATN","SNGS"]}
    tickers = presets.get(text, [t.strip() for t in re.split(r'[,\s]+', text) if t.strip()])
    
    if len(tickers) < 2:
        await update.message.reply_text("❌ Минимум 2 тикера")
        return WAITING_TICKERS
    
    await update.message.reply_text(f"⏳ Оптимизирую {len(tickers)} акций...")
    result = portfolio_analyzer.optimize_portfolio(tickers)
    await update.message.reply_text(format_portfolio_result(result), parse_mode=ParseMode.MARKDOWN)
    return ConversationHandler.END


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_sessions[update.effective_chat.id] = {}
    await update.message.reply_text("""💼 **Анализ портфеля**

Введите тикеры с долями:
`SBER:30, GAZP:25, LKOH:20, YNDX:15, GMKN:10`

/cancel для отмены""", parse_mode=ParseMode.MARKDOWN)
    return WAITING_WEIGHTS


async def process_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper().strip()
    try:
        tickers, weights = [], []
        for part in re.split(r'[,\s]+', text):
            if ':' in part:
                t, w = part.split(':')
                tickers.append(t.strip())
                weights.append(float(w))
            else:
                tickers.append(part.strip())
        if not weights: weights = [100/len(tickers)]*len(tickers)
        weights = [w/sum(weights) for w in weights]
    except:
        await update.message.reply_text("❌ Формат: SBER:30, GAZP:25")
        return WAITING_WEIGHTS
    
    user_sessions[update.effective_chat.id] = {"tickers": tickers, "weights": weights}
    await update.message.reply_text("💰 Введите сумму в рублях:\nПример: `1000000`", parse_mode=ParseMode.MARKDOWN)
    return WAITING_AMOUNT


async def process_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        amount = float(update.message.text.replace(' ','').replace(',',''))
    except:
        await update.message.reply_text("❌ Введите число")
        return WAITING_AMOUNT
    
    session = user_sessions.get(update.effective_chat.id, {})
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text(f"⏳ Анализирую на {amount:,.0f}₽...")
    
    result = portfolio_analyzer.calculate_portfolio_metrics(
        session.get("tickers", []), session.get("weights", []), amount)
    await update.message.reply_text(format_metrics_result(result), parse_mode=ParseMode.MARKDOWN)
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_sessions.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Отменено")
    return ConversationHandler.END


async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    response = await get_ai_response(update.effective_chat.id,
        "Сгенерируй 1 тестовый вопрос по инвестициям (CFA/ФСФР) с 4 вариантами A/B/C/D. Без ответа.")
    try:
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    except:
        await update.message.reply_text(response)


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("🗑 История очищена")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    response = await get_ai_response(update.effective_chat.id, update.message.text)
    try:
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    except:
        await update.message.reply_text(response)


def main():
    if not TELEGRAM_BOT_TOKEN or not EMERGENT_LLM_KEY:
        logger.error("Missing TELEGRAM_BOT_TOKEN or EMERGENT_LLM_KEY")
        return
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("stocks", stocks))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("optimize", optimize)],
        states={WAITING_TICKERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_tickers)]},
        fallbacks=[CommandHandler("cancel", cancel)]))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("analyze", analyze)],
        states={WAITING_WEIGHTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_weights)],
                WAITING_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_amount)]},
        fallbacks=[CommandHandler("cancel", cancel)]))
    app.add_handler(CommandHandler("quiz", quiz))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start", "Начать"), BotCommand("help", "Справка"),
            BotCommand("stocks", "Акции MOEX"), BotCommand("price", "Цена акции"),
            BotCommand("optimize", "Оптимизация"), BotCommand("analyze", "Анализ портфеля"),
            BotCommand("quiz", "Тест"), BotCommand("clear", "Очистить")])
    
    app.post_init = post_init
    logger.info("🚀 Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
