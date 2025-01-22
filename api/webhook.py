from fastapi import FastAPI, Request, Response
import json
import os
import sys
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your bot's modules
from treeoflifebot import (
    tree, bday, vibe, num, menu_handler, handle_photo, handle_messages,
    converter  # Import the converter instance
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    logger.error("No TELEGRAM_TOKEN found in environment variables")
    raise ValueError("No TELEGRAM_TOKEN found in environment variables")

WEBHOOK_URL = os.getenv('WEBHOOK_URL')
if not WEBHOOK_URL:
    logger.error("No WEBHOOK_URL found in environment variables")
    raise ValueError("No WEBHOOK_URL found in environment variables")

logger.info(f"Initializing bot with webhook URL: {WEBHOOK_URL}")

# Initialize the application
try:
    application = Application.builder().token(TOKEN).build()
    logger.info("Bot application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize bot application: {str(e)}", exc_info=True)
    raise

# Add handlers
try:
    application.add_handler(CommandHandler("tree", tree))
    application.add_handler(CommandHandler("bday", bday))
    application.add_handler(CommandHandler("vibe", vibe))
    application.add_handler(CommandHandler("num", num))
    application.add_handler(CallbackQueryHandler(menu_handler))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_messages))
    logger.info("All handlers registered successfully")
except Exception as e:
    logger.error(f"Failed to register handlers: {str(e)}", exc_info=True)
    raise

# Create FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    try:
        # Test bot connection
        me = await application.bot.get_me()
        logger.info(f"Bot connected successfully. Username: {me.username}, ID: {me.id}")
    except Exception as e:
        logger.error(f"Failed to connect to bot: {str(e)}", exc_info=True)
        raise

@app.post("/api/webhook")
async def webhook(request: Request):
    """Handle incoming webhook updates."""
    try:
        # Get the request body
        body = await request.json()
        logger.info(f"Received update: {body}")
        
        # Process the update
        update = Update.de_json(body, application.bot)
        if not update:
            logger.error("Failed to parse update")
            return {"status": "error", "message": "Invalid update format"}
            
        logger.info(f"Processing update type: {update.message and 'message' or update.callback_query and 'callback_query' or 'other'}")
        await application.process_update(update)
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/webhook")
async def health_check():
    """Health check endpoint."""
    try:
        # Test bot connection
        me = await application.bot.get_me()
        return {
            "status": "ok",
            "message": "Webhook is running",
            "bot_info": {
                "username": me.username,
                "id": me.id
            },
            "webhook_url": WEBHOOK_URL
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)} 