# Sacred Tree of Life Bot 🌟

A Telegram bot that combines spiritual tools with modern technology, offering Celtic tree readings, numerology calculations, ASCII art conversion, and frequency healing analysis.

## Features

- 🌳 **Tree of Life Reading**: Discover your Celtic tree personality based on your birth date
- 🔮 **Numerology Reading**: Calculate your Life Path and Destiny numbers
- 🎨 **ASCII Art Converter**: Transform images into spiritual ASCII art with customizable settings
- 🎵 **Frequency Healing**: Find your healing frequency through a personalized test

## Commands

- `/tree` - Start the bot and show main menu
- `/bday` - Get your Celtic tree personality
- `/num` - Calculate your numerology numbers
- `/vibe` - Take the frequency healing test

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tree-of-life-bot.git
cd tree-of-life-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
- Create a `.env` file in the root directory
- Add your Telegram bot token:
```
TELEGRAM_TOKEN=your_bot_token_here
```

## Deployment on Vercel

1. Fork this repository
2. Create a new project on Vercel
3. Connect your GitHub repository
4. Add the following environment variables in Vercel:
   - `TELEGRAM_TOKEN`: Your bot token
   - `WEBHOOK_URL`: Your Vercel app URL (after first deployment)
5. Deploy!

After deployment, set up your webhook:
```
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook?url=https://your-vercel-app-name.vercel.app/api/webhook
```

## Local Development

Run the bot locally:
```bash
python treeoflifebot.py
```

## Project Structure

- `treeoflifebot.py` - Main bot logic and command handlers
- `ascii_converter.py` - ASCII art conversion functionality
- `api/webhook.py` - Webhook handler for Vercel deployment
- `vercel.json` - Vercel configuration file

## Contributing

Feel free to open issues or submit pull requests with improvements!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 