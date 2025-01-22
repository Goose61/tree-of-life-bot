from fastapi import FastAPI, Request, Response
import json
import os
import sys
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

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

# Initialize bot with webhook
TOKEN = "7609264045:AAFMBSKKNAayPuiFnDhu4WCL3AbpG8a24ZI"
URL = "https://your-vercel-app-name.vercel.app"  # Replace with your Vercel app URL

# Initialize the application
application = Application.builder().token(TOKEN).build()

# Add handlers
application.add_handler(CommandHandler("tree", tree))
application.add_handler(CommandHandler("bday", bday))
application.add_handler(CommandHandler("vibe", vibe))
application.add_handler(CommandHandler("num", num))
application.add_handler(CallbackQueryHandler(menu_handler))
application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_messages))

# Create FastAPI app
app = FastAPI()

@app.post("/api/webhook")
async def webhook(request: Request):
    """Handle incoming webhook updates."""
    try:
        # Get the request body
        body = await request.json()
        
        # Process the update
        update = Update.de_json(body, application.bot)
        await application.process_update(update)
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/webhook")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Webhook is running"} 