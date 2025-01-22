from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import os
from ascii_converter import ASCIIArtConverter
import logging

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize ASCII converter with improved settings
converter = ASCIIArtConverter(
    contrast=1.5,      # Reduced contrast for better balance
    brightness=1.2,    # Slightly increased brightness
    invert=False,
    color_mode='none',
    true_color=False,
    num_threads=8,     # Use 8 threads for parallel processing
    chunk_size=25,     # Smaller chunks for better parallelization
    use_multiprocessing=False  # Avoid pickling issues
)

def get_settings_keyboard(width="Original"):
    """Create inline keyboard with settings buttons."""
    keyboard = [
        [
            InlineKeyboardButton("Invert: " + ("ON" if converter.invert else "OFF"),
                               callback_data='toggle_invert')
        ],
        [
            InlineKeyboardButton("Color Mode: " + converter.color_mode.upper(),
                               callback_data='cycle_color'),
            InlineKeyboardButton("True Color: " + ("ON" if converter.true_color else "OFF"),
                               callback_data='toggle_true_color')
        ],
        [
            InlineKeyboardButton(f"Width: {width}",
                               callback_data='set_width')
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    width = context.user_data.get('width', 200)
    keyboard = get_settings_keyboard(width)
    await update.message.reply_text(
        'Welcome to the ASCII Art Bot! 🎨\n\n'
        'Send me any image or reply to this message with an image to convert it to ASCII art.\n'
        'Use the buttons below to adjust settings before sending your image:',
        reply_markup=keyboard
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    width = context.user_data.get('width', 200)
    keyboard = get_settings_keyboard(width)
    await update.message.reply_text(
        'How to use:\n'
        '1. Send me any image or reply to the settings message with an image\n'
        '2. I will convert it to ASCII art and send you the result as an image\n\n'
        'Current settings:\n'
        'Reply to this message with an image to use these settings:',
        reply_markup=keyboard
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses."""
    query = update.callback_query
    await query.answer()

    if query.data == 'toggle_invert':
        converter.invert = not converter.invert
    elif query.data == 'cycle_color':
        if converter.true_color:
            # If true color is on, disable it first
            converter.true_color = False
        # Color modes: none -> red -> green -> blue -> yellow -> purple -> none
        modes = ['none', 'red', 'green', 'blue', 'yellow', 'purple']
        current_index = modes.index(converter.color_mode)
        converter.color_mode = modes[(current_index + 1) % len(modes)]
    elif query.data == 'toggle_true_color':
        converter.true_color = not converter.true_color
        if converter.true_color:
            # If enabling true color, set color mode to none
            converter.color_mode = 'none'
    elif query.data == 'set_width':
        keyboard = [
            [
                InlineKeyboardButton("Original (no ASCII)", callback_data='width_original'),
                InlineKeyboardButton("150", callback_data='width_150'),
                InlineKeyboardButton("200", callback_data='width_200')
            ],
            [
                InlineKeyboardButton("250", callback_data='width_250'),
                InlineKeyboardButton("300", callback_data='width_300'),
                InlineKeyboardButton("350", callback_data='width_350')
            ],
            [InlineKeyboardButton("Back to Settings", callback_data='back_to_settings')]
        ]
        await query.edit_message_text(
            "Select ASCII art width:\n"
            "'Original (no ASCII)' maintains the uploaded image resolution\n"
            "Higher values = more detail, smaller text\n\n"
            "Reply to this message with an image to use the selected width!",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
    elif query.data.startswith('width_'):
        if query.data == 'width_original':
            width = None
        else:
            width = int(query.data.split('_')[1])
        context.user_data['width'] = width if width is not None else 200  # Set to 200 if None
    elif query.data == 'back_to_settings':
        pass

    # Update the keyboard with current settings
    width = context.user_data.get('width', 200)  # Default to 200
    width_display = "Original (no ASCII)" if width is None else str(width)
    keyboard = get_settings_keyboard(width_display)
    await query.edit_message_text(
        'Current Settings:\n'
        f'Invert: {"ON" if converter.invert else "OFF"}\n'
        f'Color Mode: {converter.color_mode.upper()}\n'
        f'True Color: {"ON" if converter.true_color else "OFF"}\n'
        f'Width: {width_display}\n\n'
        '📸 Reply to this message with an image to use these settings!',
        reply_markup=keyboard
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos and convert them to ASCII art."""
    photo_path = None
    image_path = None
    processing_msg = None
    
    try:
        # Create directories if they don't exist
        os.makedirs("downloads", exist_ok=True)
        os.makedirs("ascii_output", exist_ok=True)

        # Get the photo file
        photo = await update.message.photo[-1].get_file()
        
        # Create unique filename
        chat_id = update.message.chat_id
        timestamp = int(update.message.date.timestamp())
        photo_path = os.path.join("downloads", f"photo_{chat_id}_{timestamp}.jpg")
        image_path = os.path.join("ascii_output", f"photo_{chat_id}_{timestamp}_ascii.png")

        # Send initial processing message
        processing_msg = await update.message.reply_text(
            "🎨 Processing your image...\n"
            "Converting to ASCII art with current settings..."
        )

        # Download the photo
        await photo.download_to_drive(photo_path)

        if not os.path.exists(photo_path):
            raise FileNotFoundError("Failed to download the photo")

        # Get settings
        if update.message.reply_to_message:
            width = context.user_data.get('width', 200)  # Default to 200
            width = 200 if width is None else width  # Ensure None becomes 200
        else:
            converter.invert = False
            converter.color_mode = 'none'
            converter.true_color = False
            width = 200
            context.user_data['width'] = width

        try:
            # Convert to ASCII art
            _, ascii_image = converter.image_to_ascii(
                photo_path,
                width=width,
                save_text=False,
                save_image=True,
                output_dir="ascii_output",
                output_image_path=image_path
            )

            # Verify the image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError("ASCII art image was not created successfully")

            # First, send the settings message
            width_display = "Original (no ASCII)" if width is None else str(width)
            await update.message.reply_text(
                "✨ Current Settings:\n"
                f"• Width: {width_display}\n"
                f"• Invert: {'ON' if converter.invert else 'OFF'}\n"
                f"• Color Mode: {converter.color_mode.upper()}\n"
                f"• True Color: {'ON' if converter.true_color else 'OFF'}"
            )

            # Then, send the ASCII art image
            with open(image_path, 'rb') as image_file:
                await update.message.reply_photo(
                    photo=image_file,
                    caption="🎨 Here's your ASCII art!"
                )

            # Finally, show settings keyboard for next conversion
            keyboard = get_settings_keyboard(width if width else "Original")
            await update.message.reply_text(
                "Want to try different settings? Use the buttons below and reply with an image:",
                reply_markup=keyboard
            )

        except Exception as e:
            logger.error(f"Error during ASCII conversion: {str(e)}")
            raise Exception(f"Failed to convert image to ASCII: {str(e)}")

    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        if processing_msg:
            await processing_msg.edit_text(f"❌ Sorry, there was a problem with the file: {str(e)}")
        else:
            await update.message.reply_text(f"❌ Sorry, there was a problem with the file: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        if processing_msg:
            await processing_msg.edit_text(f"❌ Sorry, an error occurred: {str(e)}")
        else:
            await update.message.reply_text("❌ Sorry, an error occurred while processing your image.")

    finally:
        # Clean up files
        try:
            if processing_msg:
                await processing_msg.delete()
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Start the bot."""
    # Create the Application and pass it your bot's token
    token = "7234888876:AAGYtnVFEijlRPhPUVtLBtxERB0bdr441GI"
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Start the Bot
    print("Starting bot...")
    print("Send a message to your bot on Telegram!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 