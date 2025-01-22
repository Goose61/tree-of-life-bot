import logging
from datetime import datetime
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from ascii_converter import ASCIIArtConverter
import os
from PIL import Image

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize ASCII converter with improved settings
converter = ASCIIArtConverter(
    contrast=1.0,
    brightness=1.0,
    invert=False,
    color_mode='none',
    true_color=True,
    num_threads=8,
    chunk_size=25,
    use_multiprocessing=False,
    use_colored_chars=False
)

# Celtic tree calendar data
TREE_CALENDAR = {
    ((1, 1), (1, 11)): "Fir Tree - The Mysterious\n\nExtraordinary taste, dignity, sophisticated, loves anything beautiful, moody, stubborn, tends to egoism but cares for those close to them, rather modest, very ambitious, talented, industrious, uncontented lover, many friends, many foes, very reliable",
    ((1, 12), (1, 24)): "Elm Tree - The Noble-mindedness\n\nPleasant shape, tasteful clothes, modest demands, tends not to forgive mistakes, cheerful, likes to lead but not to obey, honest and faithful partner, likes making decisions for others, noble minded, generous, good sense of humor, practical",
    ((1, 25), (2, 3)): "Cypress Tree - The Faithfulness\n\nStrong, muscular, adaptable, takes what life has to give, content, optimistic, craves money and acknowledgment, hates loneliness, passionate lover which cannot be satisfied, faithful, quick-tempered, unruly, pedantic, and careless",
    ((2, 4), (2, 8)): "Poplar Tree - The Uncertainty\n\nLooks very decorative, not very self-confident, only courageous if necessary, needs goodwill and pleasant surroundings, very choosy, often lonely, great animosity, artistic nature, good organizer, tends to lean toward philosophy, reliable in any situation, takes partnership seriously",
    ((2, 9), (2, 18)): "Cedar Tree - The Confidence\n\nOf rare beauty, knows how to adapt, likes luxury, of good health, not in the least shy, tends to look down on others, self-confident, determined, impatient, likes to impress others, many talents, industrious, healthy optimism, waiting for the one true love, able to make quick decisions",
    ((2, 19), (2, 28)): "Pine Tree - The Particular\n\nLoves agreeable company, very robust, knows how to make life comfortable, very active, natural, good companion, but seldom friendly, falls easily in love but its passion burns out quickly, gives up easily, everything disappointments until it finds its ideal, trustworthy, practical",
    ((3, 1), (3, 10)): "Weeping Willow - The Melancholy\n\nBeautiful, but full of melancholy, attractive, very empathetic, loves anything beautiful and tasteful, loves to travel, dreamer, restless, capricious, honest, can be influenced but is not easy to live with, demanding, good intuition, suffers in love but finds sometimes an anchoring partner",
    ((3, 11), (3, 20)): "Lime Tree - The Doubt\n\nAccepts what life is, hates fighting, stress, and labor, dislikes laziness and idleness, soft and relenting, makes sacrifices for friends, many talents but not tenacious enough to make them blossom, often wailing and complaining, very jealous but loyal",
    ((3, 21), (3, 21)): "Oak Tree - The Brave\n\nRobust nature, courageous, strong, unrelenting, independent, sensible, does not like change, keeps its feet on the ground, person of action",
    ((3, 22), (3, 31)): "Hazelnut Tree - The Extraordinary\n\nCharming, undemanding, very understanding, knows how to make an impression, active fighter for social cause, popular, moody, and capricious lover, honest, and tolerant partner, precise sense of judgment",
    ((4, 1), (4, 10)): "Rowan Tree - The Sensitivity\n\nFull of charm, cheerful, gifted without egoism, likes to draw attention, loves life, motion, unrest, and even complications, is both dependent and independent, good taste, artistic, passionate, emotional, good company, does not forgive",
    ((4, 11), (4, 20)): "Maple Tree - The Independence of Mind\n\nNo ordinary person, full of imagination and originality, shy and resented, ambitious, proud, self-confident, hungers for new experiences, sometimes nervous, has many complexities, good memory, learns easily, complicated love life, wants to impress",
    ((4, 21), (4, 30)): "Walnut Tree - The Passion\n\nUnrelenting, strange and full of contrasts, often egotistic, aggressive, noble, broad horizon, unexpected reactions, spontaneous, unlimited ambition, no flexibility, difficult and uncommon partner, not always liked but often admired, ingenious strategist, very jealous and passionate, no compromise",
    ((5, 1), (5, 14)): "Poplar Tree - The Uncertainty\n\nLooks very decorative, not very self-confident, only courageous if necessary, needs goodwill and pleasant surroundings, very choosy, often lonely, great animosity, artistic nature, good organizer, tends to lean toward philosophy, reliable in any situation, takes partnership seriously",
    ((5, 15), (5, 24)): "Chestnut Tree - The Honesty\n\nOf unusual beauty, does not want to impress, well-developed sense of justice, vivacious, interested, a born diplomat, but irritates easily and sensitive in company, often due to a lack of self-confidence, acts sometimes superior, feels not understood, loves only once, has difficulties in finding a partner",
    ((5, 25), (6, 3)): "Ash Tree - The Ambition\n\nUncommonly attractive, vivacious, impulsive, demanding, does not care for criticism, ambitious, intelligent, talented, likes to play with fate, can be egotistic, very reliable and trustworthy, faithful and prudent lover, sometimes brains rule over the heart, but takes partnership very seriously",
    ((6, 4), (6, 13)): "Hornbeam Tree - The Good Taste\n\nOf cool beauty, cares for its looks and condition, good taste, is not egoistic, makes life as comfortable as possible, leads a reasonable and disciplined life, looks for kindness and acknowledgement in an emotional partner, dreams of unusual lovers, is seldom happy with its feelings, mistrusts most people, is never sure of its decisions, very conscientious",
    ((6, 14), (6, 23)): "Fig Tree - The Sensibility\n\nVery strong, a bit self-willed, independent, does not allow contradiction or arguments, loves life, its Family, children and animals, a bit of a social butterfly, good sense of humor, likes idleness and laziness, of practical talent and intelligence",
    ((6, 24), (6, 24)): "Birch Tree - The Inspiration\n\nVivacious, attractive, elegant, friendly, unpretentious, modest, does not like anything in excess, abhors the vulgar, loves life in nature and in calm, not very passionate, full of imagination, little ambition, creates a calm and content atmosphere",
    ((6, 25), (7, 4)): "Apple Tree - The Love\n\nOf slight build, lots of charm, appeal, and attraction, pleasant aura, flirtatious, adventurous, sensitive, always in love, wants to love and beloved, faithful and tender partner, very generous, scientific talents, lives for today, a carefree philosopher with imagination",
    ((7, 5), (7, 14)): "Fir Tree - The Mysterious\n\nExtraordinary taste, dignity, sophisticated, loves anything beautiful, moody, stubborn, tends to egoism but cares for those close to them, rather modest, very ambitious, talented, industrious, uncontented lover, many friends, many foes, very reliable",
    ((7, 15), (7, 25)): "Elm Tree - The Noble-mindedness\n\nPleasant shape, tasteful clothes, modest demands, tends not to forgive mistakes, cheerful, likes to lead but not to obey, honest and faithful partner, likes making decisions for others, noble minded, generous, good sense of humor, practical",
    ((7, 26), (8, 4)): "Cypress Tree - The Faithfulness\n\nStrong, muscular, adaptable, takes what life has to give, content, optimistic, craves money and acknowledgment, hates loneliness, passionate lover which cannot be satisfied, faithful, quick-tempered, unruly, pedantic, and careless",
    ((8, 5), (8, 13)): "Poplar Tree - The Uncertainty\n\nLooks very decorative, not very self-confident, only courageous if necessary, needs goodwill and pleasant surroundings, very choosy, often lonely, great animosity, artistic nature, good organizer, tends to lean toward philosophy, reliable in any situation, takes partnership seriously",
    ((8, 14), (8, 23)): "Cedar Tree - The Confidence\n\nOf rare beauty, knows how to adapt, likes luxury, of good health, not in the least shy, tends to look down on others, self-confident, determined, impatient, likes to impress others, many talents, industrious, healthy optimism, waiting for the one true love, able to make quick decisions",
    ((8, 24), (9, 2)): "Pine Tree - The Particular\n\nLoves agreeable company, very robust, knows how to make life comfortable, very active, natural, good companion, but seldom friendly, falls easily in love but its passion burns out quickly, gives up easily, everything disappointments until it finds its ideal, trustworthy, practical",
    ((9, 3), (9, 12)): "Weeping Willow - The Melancholy\n\nBeautiful, but full of melancholy, attractive, very empathetic, loves anything beautiful and tasteful, loves to travel, dreamer, restless, capricious, honest, can be influenced but is not easy to live with, demanding, good intuition, suffers in love but finds sometimes an anchoring partner",
    ((9, 13), (9, 22)): "Lime Tree - The Doubt\n\nAccepts what life is, hates fighting, stress, and labor, dislikes laziness and idleness, soft and relenting, makes sacrifices for friends, many talents but not tenacious enough to make them blossom, often wailing and complaining, very jealous but loyal",
    ((9, 23), (9, 23)): "Olive Tree - The Wisdom\n\nLoves sun, warmth and kind feelings, reasonable, balanced, avoids aggression and violence, tolerant, cheerful, calm, well-developed sense of justice, sensitive, empathetic, free of jealousy, loves to read and the company of sophisticated people",
    ((9, 24), (10, 3)): "Hazelnut Tree - The Extraordinary\n\nCharming, undemanding, very understanding, knows how to make an impression, active fighter for social cause, popular, moody, and capricious lover, honest, and tolerant partner, precise sense of judgment",
    ((10, 4), (10, 13)): "Rowan Tree - The Sensitivity\n\nFull of charm, cheerful, gifted without egoism, likes to draw attention, loves life, motion, unrest, and even complications, is both dependent and independent, good taste, artistic, passionate, emotional, good company, does not forgive",
    ((10, 14), (10, 23)): "Maple Tree - The Independence of Mind\n\nNo ordinary person, full of imagination and originality, shy and resented, ambitious, proud, self-confident, hungers for new experiences, sometimes nervous, has many complexities, good memory, learns easily, complicated love life, wants to impress",
    ((10, 24), (11, 11)): "Walnut Tree - The Passion\n\nUnrelenting, strange and full of contrasts, often egotistic, aggressive, noble, broad horizon, unexpected reactions, spontaneous, unlimited ambition, no flexibility, difficult and uncommon partner, not always liked but often admired, ingenious strategist, very jealous and passionate, no compromise",
    ((11, 12), (11, 21)): "Chestnut Tree - The Honesty\n\nOf unusual beauty, does not want to impress, well-developed sense of justice, vivacious, interested, a born diplomat, but irritates easily and sensitive in company, often due to a lack of self-confidence, acts sometimes superior, feels not understood, loves only once, has difficulties in finding a partner",
    ((11, 22), (12, 1)): "Ash Tree - The Ambition\n\nUncommonly attractive, vivacious, impulsive, demanding, does not care for criticism, ambitious, intelligent, talented, likes to play with fate, can be egotistic, very reliable and trustworthy, faithful and prudent lover, sometimes brains rule over the heart, but takes partnership very seriously",
    ((12, 2), (12, 11)): "Hornbeam Tree - The Good Taste\n\nOf cool beauty, cares for its looks and condition, good taste, is not egoistic, makes life as comfortable as possible, leads a reasonable and disciplined life, looks for kindness and acknowledgement in an emotional partner, dreams of unusual lovers, is seldom happy with its feelings, mistrusts most people, is never sure of its decisions, very conscientious",
    ((12, 12), (12, 21)): "Fig Tree - The Sensibility\n\nVery strong, a bit self-willed, independent, does not allow contradiction or arguments, loves life, its Family, children and animals, a bit of a social butterfly, good sense of humor, likes idleness and laziness, of practical talent and intelligence",
    ((12, 22), (12, 22)): "Beech Tree - The Creative\n\nHas good taste, concerned about its looks, materialistic, good organization of life and career, economical, good leader, takes no unnecessary risks, reasonable, splendid lifetime companion, keen on keeping fit (diets, sports, etc.)",
    ((12, 23), (12, 31)): "Apple Tree - The Love\n\nOf slight build, lots of charm, appeal, and attraction, pleasant aura, flirtatious, adventurous, sensitive, always in love, wants to love and beloved, faithful and tender partner, very generous, scientific talents, lives for today, a carefree philosopher with imagination"
}

# Frequency healing test data
FREQUENCY_QUESTIONS = {
    1: {
        'question': "1. When you recall a memory, how do you remember it?",
        'options': {
            'a': "The smells and aromas",
            'b': "How you felt",
            'c': "What you heard and said",
            'd': "What you saw and the scenery"
        }
    },
    2: {
        'question': "2. What health benefits are you seeking?",
        'options': {
            'a': "Relief from depression, anxiety, and/or stress",
            'b': "Elimination of body aches and pain",
            'c': "Reduction of energetic blockages and limiting beliefs",
            'd': "To feel more grounded and aligned"
        }
    },
    3: {
        'question': "3. How do you want to feel?",
        'options': {
            'a': "Peaceful",
            'b': "Healed",
            'c': "Confident",
            'd': "Balanced"
        }
    },
    4: {
        'question': "4. What low-level emotions do you feel most often or have you felt most in your life?",
        'options': {
            'a': "Worry/fear",
            'b': "Shame/guilt/sadness",
            'c': "Anger/grief",
            'd': "Frustration/dissatisfaction"
        }
    },
    5: {
        'question': "5. What is your idea of the perfect activity or day?",
        'options': {
            'a': "Going to a rose garden, a lavender field, or perfume shopping",
            'b': "Getting a massage",
            'c': "Going to a concert and seeing your favorite musician play",
            'd': "Going to the mountains, the ocean, or otherwise relaxing in nature"
        }
    }
}

FREQUENCY_RESULTS = {
    'a': """🌸 Emotional Frequency (396 Hz)
You are deeply connected to emotional healing and processing. The 396 Hz frequency can help liberate you from fear and guilt, transforming grief into joy and negative emotions into positive ones.

Your vibrational healing style is Aromatherapy.

Whether you're looking for peace of mind, mental clarity, stress relief, or to balance your mood and body, you love breathing in the aromas of life and they can be very healing for you. You can benefit from aromatherapy, using the natural or essential oils extracted from flowers, bark, stems, leaves, roots, or other parts of a plant to enhance psychological and physical wellbeing.

When it comes to vibrational healing, the essential oils in aromatherapy activate the smell receptors in your nose, which send messages directly to your brain's limbic system. Inhaling essential oils triggers a pleasant "natural relaxation response" that helps you sleep better, reduces stress and joint pain, and even improves blood circulation. Aromatherapy can be used in the following ways: direct inhalation, through a vaporizer or humidifier, in a perfume, and with aromatherapy diffusers.""",

    'b': """✨ Physical Frequency (417 Hz)
You are attuned to physical healing and transformation. The 417 Hz frequency facilitates change and helps clear traumatic experiences from the body.

Your vibrational healing style involves Emotional Energy Release Techniques.

If you want to balance your emotions, free yourself from pain and limiting beliefs, and bring your body into harmony, you might thrive with Emotional Release Techniques, including acupuncture, energy medicine, tapping, and massage. These methods focus on the energetic vibrations of your body and aim to harmonize them, creating a powerful holistic healing technique that effectively resolves a range of issues, including stress, anxiety, phobias, emotional disorders, chronic pain, weight control, and limiting beliefs.

You might enjoy tapping, which is based on the combination of ancient Chinese acupressure and modern psychology. The practice consists of tapping with your fingertips on specific meridian points while focusing on negative emotions or physical sensations. Doing this helps calm the nervous system, rewire the brain to respond in healthier ways, and restore the body's balance of energy. You may also enjoy network spinal analysis (NSA), where a network practitioner provides specific, gentle touch on the spine to update and cue the nervous system and improve quality of life. NSA can help to release tension and trauma stored in the spine and nervous system. All of the emotional energetic release vibrational healing modalities focus on bringing you into balance by using the energetic meridians on your body to initiate innate self-healing abilities.""",

    'c': """🎵 Spiritual Frequency (528 Hz)
You resonate with spiritual awakening and DNA repair. The 528 Hz frequency is known as the 'miracle tone' and can help with clarity, peace, and spiritual development.

Your vibrational healing style is Sound Healing.

Have you ever been in a funky mood and put on a favorite song and noticed that you started to feel better? That's the effect of sound healing. But it's more than just listening to good music—it's an ancient healing technique that uses tonal frequencies to bring the body into a state of vibrational balance and harmony. It balances and clears the mind, and leads to a renewed sense of purpose, wellbeing, calm, and focus.

Sound healing therapies, also known as sound baths, utilize singing bowls, gongs, Tibetan bowls, vocal chanting, tuning forks, and drums. These work because the frequencies slow down your brain waves to a deeply restorative state, which activates the body's system of self-healing.

You'll find great harmony with sound healing, as you "bathe" in the soothing sounds and vibrations.""",

    'd': """🌿 Grounding Frequency (639 Hz)
You connect strongly with earth energy and relationship healing. The 639 Hz frequency helps balance relationships and promotes connection with nature.

Your vibrational healing style is Earth Energy.

The earth, Gaia, has profound wisdom, natural grounding vibrations, and healing capacities. When we align with elements from the earth, we bring our bodies into ultimate harmony and alignment.

Crystals are one earth energy option, which healers, shamans, and priests have long used for their unique and special properties. Crystals are set on and around the body, which aids in drawing out any negative energy. You can also use crystals in your daily life by working with their energetic vibration. Because they come from the earth, they inspire a grounded healing process that helps balance your body, mind and soul. It is widely believed crystals vibrate at the same pitch as humans—such that the resonance between the stone and the human either combats the vibration of the illness or amplifies that of health. You can work with crystal energy to amplify and balance the energy in the room and to assist in channeling universal life energy.

Flower essences provide another earth energy option. Botanical herbs and flowering plants have been used since the time of ancient Egypt and in many indigenous communities for healing emotional imbalances. Flower essences, also known as flower remedies, are bioenergetic remedies containing the unique energetic vibration of plants, which have healing effects on mental-emotional wellbeing. Not only can they help you heal negative emotional patterns, move through energetic blockages, and better manage everyday life stressors, but restoring inner calm and balance on the mental, emotional, and energetic levels frees up resources and energy for the physical body to heal.

Finally, grounding is a good healing practice for this style of spiritual vibration healing. The earth has an infinite supply of free electrons, so when a person is connected to it and grounded, walking barefoot outside or sitting in the grass, those electrons naturally flow between the earth and the human body, reducing free radicals and eliminating any static electrical charge. Just like the sun constantly provides us with important energy, nutrients, and vitamins, the earth is a source of subtle energy that provides important elements that contribute to optimum health."""
}

# Numerology data
NUMEROLOGY_MEANINGS = {
    1: """Number 1 - The Leader
You're independent, creative, and a natural-born leader. You have strong ambitions and the determination to achieve them. You're innovative and original in your thinking, often pioneering new paths.""",
    
    2: """Number 2 - The Mediator
You're diplomatic, cooperative, and sensitive to others' needs. You have natural abilities in partnership and mediation. You bring harmony and balance to situations.""",
    
    3: """Number 3 - The Creative
You're expressive, optimistic, and naturally creative. You have a gift for communication and bringing joy to others. You're socially engaging and have a natural charm.""",
    
    4: """Number 4 - The Builder
You're practical, organized, and hardworking. You have a strong foundation of values and work ethic. You're reliable and create stability wherever you go.""",
    
    5: """Number 5 - The Freedom Seeker
You're adventurous, versatile, and progressive. You love freedom and new experiences. You adapt easily to change and inspire others to explore.""",
    
    6: """Number 6 - The Nurturer
You're responsible, caring, and protective. You have a deep sense of service and duty to others. You're a natural caretaker with strong family values.""",
    
    7: """Number 7 - The Seeker
You're analytical, introspective, and spiritual. You have a keen mind for investigation and research. You seek deeper meaning in everything.""",
    
    8: """Number 8 - The Powerhouse
You're ambitious, successful, and materially focused. You have natural leadership abilities in business and finance. You're driven to achieve great things.""",
    
    9: """Number 9 - The Humanitarian
You're compassionate, selfless, and wise. You have a strong sense of global consciousness. You're here to serve humanity and make a difference.""",
    
    11: """Master Number 11 - The Intuitive
You have heightened intuitive abilities and spiritual insight. You're here to heal yourself and others through your elevated psychic abilities. You often experience extreme life circumstances that develop your extrasensory talents.""",
    
    22: """Master Number 22 - The Master Builder
You have the potential to achieve great things on a large scale. You can turn dreams into reality and create lasting structures. You're innovative and capable of transforming society."""
}

LETTER_VALUES = {
    'A': 1, 'J': 1, 'S': 1,
    'B': 2, 'K': 2, 'T': 2,
    'C': 3, 'L': 3, 'U': 3,
    'D': 4, 'M': 4, 'V': 4,
    'E': 5, 'N': 5, 'W': 5,
    'F': 6, 'O': 6, 'X': 6,
    'G': 7, 'P': 7, 'Y': 7,
    'H': 8, 'Q': 8, 'Z': 8,
    'I': 9, 'R': 9
}

def calculate_life_path(birth_date: datetime) -> int:
    """Calculate Life Path Number from birth date."""
    month = birth_date.month
    day = sum(int(d) for d in str(birth_date.day))
    year = sum(int(d) for d in str(birth_date.year))
    
    # First reduction
    month = sum(int(d) for d in str(month))
    day = sum(int(d) for d in str(day))
    year = sum(int(d) for d in str(year))
    
    # Final reduction
    total = month + day + year
    if total in [11, 22]:  # Master Numbers
        return total
    
    while total > 9:
        total = sum(int(d) for d in str(total))
    
    return total

def calculate_destiny(full_name: str) -> int:
    """Calculate Destiny Number from full name."""
    # Remove spaces and convert to uppercase
    name = full_name.upper().replace(" ", "")
    
    # Calculate total
    total = sum(LETTER_VALUES.get(letter, 0) for letter in name)
    
    # Reduce to single digit or master number
    while total > 9 and total not in [11, 22]:
        total = sum(int(d) for d in str(total))
    
    return total

async def num(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the numerology command."""
    # Delete the command message for privacy
    try:
        await update.message.delete()
    except Exception:
        pass

    # Store user ID and message ID for later deletion
    context.user_data['numerology_request'] = {
        'user_id': update.effective_user.id,
        'question_message_id': None
    }
    
    sent_message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🔮 Welcome to the Numerology Calculator! 🔮\n\n"
            "Please use one of these formats:\n"
            "1. Full reading: DD/MM/YYYY, Full Name or Username\n"
            "2. Life Path only: DD/MM/YYYY\n"
            "3. Destiny only: Your Full Name or Username\n"
            "Examples:\n"
            "25/12/1990, Andy Chad Ayrey\n"
            "25/12/1990\n"
            "Andy Chad Ayrey\n\n"
             "🔒 For privacy, your response will be deleted immediately after processing."
    )
    context.user_data['numerology_request']['question_message_id'] = sent_message.message_id

async def handle_numerology(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process numerology calculation request."""
    try:
        # Check if this is a response to a numerology request
        if 'numerology_request' not in context.user_data or \
           context.user_data['numerology_request'].get('user_id') != update.effective_user.id or \
           not update.message.reply_to_message or \
           update.message.reply_to_message.from_user.id != context.bot.id or \
           not any(prompt in update.message.reply_to_message.text 
                  for prompt in ["Numerology Calculator",
                               "please provide both your birth date"]):
            return
            
        # Store message data before deletion
        text = update.message.text
        chat_id = update.effective_chat.id

        # Immediately delete the user's response for privacy
        try:
            await update.message.delete()
            await update.message.reply_to_message.delete()
        except Exception:
            pass

        # Check input format
        if ',' in text:  # Full reading
            date_part, name_part = text.split(',', 1)
            day, month, year = map(int, date_part.strip().split('/'))
            birth_date = datetime(year, month, day)
            
            # Calculate numbers
            life_path = calculate_life_path(birth_date)
            destiny = calculate_destiny(name_part.strip())
            
            # Delete the question message
            if context.user_data['numerology_request']['question_message_id']:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=context.user_data['numerology_request']['question_message_id']
                    )
                except Exception:
                    pass
            
            # Send full results
            response = f"🔮 Your Numerology Reading 🔮\n\n"
            response += f"Life Path Number: {life_path}\n"
            response += f"{NUMEROLOGY_MEANINGS[life_path]}\n\n"
            response += f"Destiny Number: {destiny}\n"
            response += f"{NUMEROLOGY_MEANINGS[destiny]}\n\n"
            response += "Remember: These numbers reveal your greater purpose and how you will express your goals in life."
        elif '/' in text:  # Life Path only
            day, month, year = map(int, text.strip().split('/'))
            birth_date = datetime(year, month, day)
            
            # Calculate life path number
            life_path = calculate_life_path(birth_date)
            
            # Delete the question message
            if context.user_data['numerology_request']['question_message_id']:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=context.user_data['numerology_request']['question_message_id']
                    )
                except Exception:
                    pass
            
            # Send life path results only
            response = f"🔮 Your Life Path Number Reading 🔮\n\n"
            response += f"Life Path Number: {life_path}\n"
            response += f"{NUMEROLOGY_MEANINGS[life_path]}\n\n"
            response += "To get your full numerology reading including your Destiny Number, please provide both your birth date and full name."
        else:  # Destiny only
            # Calculate destiny number from name
            destiny = calculate_destiny(text.strip())
            
            # Delete the question message
            if context.user_data['numerology_request']['question_message_id']:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=context.user_data['numerology_request']['question_message_id']
                    )
                except Exception:
                    pass
            
            # Send destiny results only
            response = f"🔮 Your Destiny Number Reading 🔮\n\n"
            response += f"Destiny Number: {destiny}\n"
            response += f"{NUMEROLOGY_MEANINGS[destiny]}\n\n"
            response += "To get your full numerology reading including your Life Path Number, please provide both your birth date and full name."
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=response
        )
        
        # Clear numerology request data
        del context.user_data['numerology_request']
        
    except ValueError:
        error_msg = await update.message.reply_text(
            "Please use one of these formats:\n"
            "1. Full reading: DD/MM/YYYY, Full Name\n"
            "2. Life Path only: DD/MM/YYYY\n"
            "3. Destiny only: Your Full Name\n"
            "Examples:\n"
            "25/12/1990, Andy Chad Ayrey\n"
            "25/12/1990\n"
            "Andy Chad Ayrey"
        )
        await asyncio.sleep(3)
        await error_msg.delete()
    except Exception as e:
        error_msg = await update.message.reply_text(
            "Sorry, there was an error processing your request. Please try again."
        )
        await asyncio.sleep(3)
        await error_msg.delete()

def get_tree(birth_date: datetime) -> str:
    month, day = birth_date.month, birth_date.day
    
    for (start_month, start_day), (end_month, end_day) in TREE_CALENDAR.keys():
        # If start and end months are the same
        if start_month == end_month:
            if month == start_month and start_day <= day <= end_day:
                return TREE_CALENDAR[((start_month, start_day), (end_month, end_day))]
        # If the date range spans across months
        elif (month == start_month and day >= start_day) or \
             (month == end_month and day <= end_day) or \
             (start_month == 12 and month == 1 and day <= end_day) or \
             (month > start_month and month < end_month):
            return TREE_CALENDAR[((start_month, start_day), (end_month, end_day))]
    
    return "Date not found in Tree of lIfe tree calendar"

async def bday(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if date was provided with command
    if context.args:
        try:
            date_text = context.args[0]
            day, month = map(int, date_text.split('/'))
            birth_date = datetime(2000, month, day)  # Year doesn't matter for this purpose
            tree = get_tree(birth_date)
            await update.message.reply_text(
                f"Based on your birthday ({day}/{month}), your Tree of Life tree personality is:\n\n"
                f"🌳 {tree} 🌳"
            )
        except (ValueError, IndexError):
            await update.message.reply_text(
                "Please use the correct format: /bday DD/MM (e.g., /bday 25/12) or reply with just DD/MM"
            )
    else:
        await update.message.reply_text(
            "🌳 Welcome to the Sacred Tree of Life Personality Bot! 🌳\n"
            "Please reply to this message with your birthday in DD/MM format (e.g., 25/12 for December 25th)\n"
            "Or use the command format: /bday DD/MM (e.g., /bday 25/12)"
        )

async def handle_birthday(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle birthday input for tree personality."""
    # Check if the message is a reply to the bot's birthday prompt
    if not update.message.reply_to_message or \
       update.message.reply_to_message.from_user.id != context.bot.id or \
       not any(prompt in update.message.reply_to_message.text 
              for prompt in ["Please reply to this message with your birthday",
                           "send your birthday in the correct format"]):
        return
        
    try:
        # Parse the date
        day, month = map(int, update.message.text.split('/'))
        birth_date = datetime(2000, month, day)  # Year doesn't matter for this purpose
        
        # Get the corresponding tree
        tree = get_tree(birth_date)
        
        # Delete the user's message and the prompt for cleanliness
        try:
            await update.message.delete()
            await update.message.reply_to_message.delete()
        except Exception:
            pass
        
        # Send the result with a return to main menu button
        main_menu_button = [[InlineKeyboardButton("🏠 Main Menu", callback_data='return_main')]]
        await update.message.reply_text(
            f"Based on your birthday ({day}/{month}), your Tree of Life tree personality is:\n\n"
            f"🌳 {tree} 🌳",
            reply_markup=InlineKeyboardMarkup(main_menu_button)
        )
    except ValueError:
        await update.message.reply_text(
            "Please send your birthday in the correct format: DD/MM (e.g., 25/12 for December 25th)"
        )
    except Exception as e:
        await update.message.reply_text(
            "Sorry, there was an error processing your birthday. Please try again."
        )

def interpret_frequency_answers(answers: dict) -> str:
    # Count the frequency of each option
    option_counts = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    for answer in answers.values():
        option_counts[answer.lower()] += 1
    
    # Find the most common answer
    dominant_frequency = max(option_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate percentages for each answer
    total_questions = len(answers)
    percentages = {
        option: (count / total_questions) * 100 
        for option, count in option_counts.items()
    }
    
    # Create the analysis text
    analysis = f"🎵 Your Frequency Healing Test Results 🎵\n\nAnswer Distribution:\n"
    for option in ['a', 'b', 'c', 'd']:
        analysis += f"{option.upper()}: {percentages[option]:.0f}%\n"
    
    analysis += f"\nBased on your answers, here is your detailed healing profile:\n\n{FREQUENCY_RESULTS[dominant_frequency]}\n\n"
    analysis += """Remember: Everything is energy. When energy is uneven or blocked in the body and aura field, it can show up as illness, dis-ease and troubling emotions. With this concept as a driving force, vibrational healing aims to activate the body's energies toward equilibrium and balance."""
    
    return analysis

async def vibe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the frequency healing test."""
    # Initialize user data for the test
    context.user_data['frequency_test'] = {
        'current_question': 1,
        'answers': {},
        'user_id': update.effective_user.id,  # Store the user ID who started the test
        'last_question_message_id': None  # Track the last question message ID
    }
    
    # Send the first question
    question_data = FREQUENCY_QUESTIONS[1]
    message = f"{question_data['question']}\n\n"
    for option, text in question_data['options'].items():
        message += f"{option}) {text}\n"
    
    sent_message = await update.message.reply_text(
        "🎵 Welcome to the Frequency Healing Test! 🎵\n"
        "Answer each question by replying with a single letter (a, b, c, or d).\n\n" + message
    )
    context.user_data['frequency_test']['last_question_message_id'] = sent_message.message_id

async def handle_frequency_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle answers for the frequency test."""
    # Check if user is in the middle of a test and is authorized
    if 'frequency_test' not in context.user_data or \
       context.user_data['frequency_test'].get('user_id') != update.effective_user.id or \
       not update.message.reply_to_message or \
       update.message.reply_to_message.from_user.id != context.bot.id:
        return

    answer = update.message.text.lower().strip()
    if answer not in ['a', 'b', 'c', 'd']:
        await update.message.reply_text("Please answer with a single letter: a, b, c, or d")
        return

    current_q = context.user_data['frequency_test']['current_question']
    context.user_data['frequency_test']['answers'][current_q] = answer

    # Delete the previous question message if it exists
    if context.user_data['frequency_test']['last_question_message_id']:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=context.user_data['frequency_test']['last_question_message_id']
            )
        except Exception:
            pass  # Ignore if message can't be deleted

    # Delete the user's answer message
    try:
        await update.message.delete()
    except Exception:
        pass  # Ignore if message can't be deleted

    # Move to next question or finish test
    if current_q < 5:
        next_q = current_q + 1
        context.user_data['frequency_test']['current_question'] = next_q
        
        question_data = FREQUENCY_QUESTIONS[next_q]
        message = f"{question_data['question']}\n\n"
        for option, text in question_data['options'].items():
            message += f"{option}) {text}\n"
        
        # Send new message instead of replying
        sent_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=message
        )
        context.user_data['frequency_test']['last_question_message_id'] = sent_message.message_id
    else:
        # Test is complete, interpret results
        result = interpret_frequency_answers(context.user_data['frequency_test']['answers'])
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=result
        )
        # Clear test data
        del context.user_data['frequency_test']

async def handle_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route messages to appropriate handlers based on context."""
    # Only process messages that are direct replies to the bot's messages
    if not update.message.reply_to_message or \
       update.message.reply_to_message.from_user.id != context.bot.id:
        return

    # Check if this is a reply to a specific prompt
    reply_text = update.message.reply_to_message.text
    
    if 'frequency_test' in context.user_data and \
       context.user_data['frequency_test'].get('user_id') == update.effective_user.id:
        await handle_frequency_answer(update, context)
    elif 'numerology_request' in context.user_data and \
         context.user_data['numerology_request'].get('user_id') == update.effective_user.id and \
         ("Numerology Calculator" in reply_text or "please provide both your birth date" in reply_text):
        await handle_numerology(update, context)
    elif "Please reply to this message with your birthday" in reply_text or \
         "send your birthday in the correct format" in reply_text:
        await handle_birthday(update, context)
    else:
        # If it's not a reply to any specific prompt, ignore it
        return

async def tree(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message with the main menu when the command /tree is issued."""
    # Store user ID and last message ID for management
    context.user_data['menu_state'] = {
        'user_id': update.effective_user.id,
        'last_message_id': None
    }
    
    main_menu_keyboard = [
        [
            InlineKeyboardButton("🌳 Tree of Life Reading", callback_data='menu_tree'),
            InlineKeyboardButton("🔮 Numerology Reading", callback_data='menu_num')
        ],
        [
            InlineKeyboardButton("🎨 ASCII Art Converter", callback_data='menu_ascii'),
            InlineKeyboardButton("🎵 Frequency Healing", callback_data='menu_vibe')
        ],
        [
            InlineKeyboardButton("ℹ️ About", callback_data='menu_about'),
            InlineKeyboardButton("❓ Help", callback_data='menu_help')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(main_menu_keyboard)
    
    welcome_message = (
        "🌟 Welcome to the Sacred Tree of Life Bot! 🌟\n\n"
        "I am your spiritual companion on a journey of self-discovery and healing.\n\n"
        "Choose from these sacred tools:\n"
        "🌳 Tree of Life Reading - Discover your Celtic tree personality\n"
        "🔮 Numerology Reading - Uncover your life path and destiny numbers\n"
        "🎨 ASCII Art Converter - Transform images into spiritual ASCII art\n"
        "🎵 Frequency Healing - Find your healing frequency\n\n"
        "Select an option below to begin your journey:"
    )
    
    # Delete the command message for cleanliness
    try:
        await update.message.delete()
    except Exception:
        pass
    
    try:
        # Try to send as a reply first
        sent_message = await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    except Exception:
        # If reply fails, send as a new message
        sent_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_message,
            reply_markup=reply_markup
        )
    
    context.user_data['menu_state']['last_message_id'] = sent_message.message_id

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle menu button clicks."""
    query = update.callback_query
    await query.answer()
    
    # Check if user is authorized to use this menu
    if 'menu_state' not in context.user_data or \
       context.user_data['menu_state']['user_id'] != update.effective_user.id:
        await query.answer("This menu can only be used by the person who initiated it.", show_alert=True)
        return
    
    # Update last interaction time for ASCII menu buttons
    if query.data.startswith('ascii_'):
        context.user_data['last_ascii_interaction'] = datetime.now().timestamp()
    
    # Initialize ascii settings if not exists
    if 'ascii_settings' not in context.user_data:
        context.user_data['ascii_settings'] = {
            'contrast': 1.0,
            'brightness': 1.0,
            'width': 250,
            'color_mode': 'none',
            'use_colored_chars': False
        }
    
    settings = context.user_data['ascii_settings']
    
    # Handle ASCII settings adjustments
    if query.data == 'ascii_contrast_up':
        settings['contrast'] = min(5.0, settings['contrast'] + 0.5)
        converter.contrast = settings['contrast']
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_contrast_down':
        settings['contrast'] = max(0.5, settings['contrast'] - 0.5)
        converter.contrast = settings['contrast']
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_brightness_up':
        settings['brightness'] = min(2.0, settings['brightness'] + 0.1)
        converter.brightness = settings['brightness']
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_brightness_down':
        settings['brightness'] = max(0.1, settings['brightness'] - 0.1)
        converter.brightness = settings['brightness']
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_width_up':
        settings['width'] = min(800, settings['width'] + 50)
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_width_down':
        settings['width'] = max(50, settings['width'] - 50)
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_toggle_invert':
        converter.invert = not converter.invert
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_cycle_color':
        if converter.true_color:
            converter.true_color = False
        modes = ['none', 'red', 'green', 'blue', 'yellow', 'purple']
        current_index = modes.index(settings['color_mode'])
        settings['color_mode'] = modes[(current_index + 1) % len(modes)]
        converter.color_mode = settings['color_mode']
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_toggle_true_color':
        converter.true_color = not converter.true_color
        if converter.true_color:
            settings['color_mode'] = 'none'
            converter.color_mode = 'none'
        await handle_ascii_menu(update, context)
    elif query.data == 'ascii_reset':
        settings['contrast'] = 1.0
        settings['brightness'] = 1.0
        settings['width'] = 250
        settings['color_mode'] = 'none'
        settings['use_colored_chars'] = False
        converter.contrast = 1.0
        converter.brightness = 1.0
        converter.color_mode = 'none'
        converter.invert = False
        converter.true_color = False
        converter.use_colored_chars = False
        await handle_ascii_menu(update, context)
    # Handle main menu buttons
    elif query.data == 'menu_tree':
        await query.message.reply_text(
            "🌳 Welcome to the Sacred Tree of Life Personality Bot! 🌳\n"
            "Please reply to this message with your birthday in DD/MM format (e.g., 25/12 for December 25th)"
        )
    elif query.data == 'menu_num':
        sent_message = await query.message.reply_text(
            "🔮 Welcome to the Numerology Calculator! 🔮\n\n"
            "Please use one of these formats:\n"
            "1. Full reading: DD/MM/YYYY, Full Name or Username\n"
            "2. Life Path only: DD/MM/YYYY\n"
            "3. Destiny only: Your Full Name or Username\n"
            "Examples:\n"
            "25/12/1990, Andy Chad Ayrey\n"
            "25/12/1990\n"
            "Andy Chad Ayrey\n\n"
            "🔒 For privacy, your response will be deleted immediately after processing."
        )
        context.user_data['numerology_request'] = {
            'user_id': update.effective_user.id,
            'question_message_id': sent_message.message_id
        }
    elif query.data == 'menu_vibe':
        context.user_data['frequency_test'] = {
            'current_question': 1,
            'answers': {},
            'user_id': update.effective_user.id,
            'last_question_message_id': None
        }
        
        question_data = FREQUENCY_QUESTIONS[1]
        message = f"{question_data['question']}\n\n"
        for option, text in question_data['options'].items():
            message += f"{option}) {text}\n"
        
        sent_message = await query.message.reply_text(
            "🎵 Welcome to the Frequency Healing Test! 🎵\n"
            "Answer each question by replying with a single letter (a, b, c, or d).\n\n" + message
        )
        context.user_data['frequency_test']['last_question_message_id'] = sent_message.message_id
    elif query.data == 'menu_ascii':
        await handle_ascii_menu(update, context)
    elif query.data == 'menu_about':
        about_text = (
            "🌟 About Sacred Tree of Life Bot 🌟\n\n"
            "This spiritual companion combines ancient wisdom with modern technology to help you on your journey of self-discovery.\n\n"
            "Features:\n"
            "🌳 Celtic Tree Calendar Readings\n"
            "🔮 Numerology Calculations\n"
            "🎨 Spiritual ASCII Art Conversion\n"
            "🎵 Frequency Healing Analysis\n\n"
            "Created with love and intention to help you connect with your higher self. 💫"
        )
        await query.edit_message_text(about_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data='return_main')]]))
    elif query.data == 'menu_help':
        help_text = (
            "🌟 How to Use Sacred Tree of Life Bot 🌟\n\n"
            "🌳 Tree Reading: Use /bday or send your birthdate (DD/MM)\n"
            "🔮 Numerology: Use /num and follow the prompts\n"
            "🎨 ASCII Art: Send any image to convert\n"
            "🎵 Frequency Test: Use /vibe and answer the questions\n\n"
            "Need more help? Feel free to ask! 💫"
        )
        await query.edit_message_text(help_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data='return_main')]]))
    elif query.data == 'return_main':
        main_menu_keyboard = [
            [
                InlineKeyboardButton("🌳 Tree of Life Reading", callback_data='menu_tree'),
                InlineKeyboardButton("🔮 Numerology Reading", callback_data='menu_num')
            ],
            [
                InlineKeyboardButton("🎨 ASCII Art Converter", callback_data='menu_ascii'),
                InlineKeyboardButton("🎵 Frequency Healing", callback_data='menu_vibe')
            ],
            [
                InlineKeyboardButton("ℹ️ About", callback_data='menu_about'),
                InlineKeyboardButton("❓ Help", callback_data='menu_help')
            ]
        ]
        welcome_message = (
            "🌟 Sacred Tree of Life Bot 🌟\n\n"
            "Choose from these sacred tools:\n"
            "🌳 Tree of Life Reading - Discover your Celtic tree personality\n"
            "🔮 Numerology Reading - Uncover your life path and destiny numbers\n"
            "🎨 ASCII Art Converter - Transform images into spiritual ASCII art\n"
            "🎵 Frequency Healing - Find your healing frequency"
        )
        await query.edit_message_text(welcome_message, reply_markup=InlineKeyboardMarkup(main_menu_keyboard))

async def handle_ascii_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle ASCII art converter menu."""
    # Get current settings from context or use defaults
    if 'ascii_settings' not in context.user_data:
        context.user_data['ascii_settings'] = {
            'contrast': 1.0,
            'brightness': 1.0,
            'width': 250,
            'color_mode': 'none',
            'use_colored_chars': False
        }
    
    settings = context.user_data['ascii_settings']
    
    # Cancel any existing menu deletion task
    if 'menu_deletion_task' in context.user_data:
        try:
            context.user_data['menu_deletion_task'].cancel()
        except Exception:
            pass
    
    ascii_menu_keyboard = [
        [
            InlineKeyboardButton("Contrast +", callback_data='ascii_contrast_up'),
            InlineKeyboardButton(f"{settings['contrast']:.1f}", callback_data='ascii_contrast_info'),
            InlineKeyboardButton("Contrast -", callback_data='ascii_contrast_down')
        ],
        [
            InlineKeyboardButton("Brightness +", callback_data='ascii_brightness_up'),
            InlineKeyboardButton(f"{settings['brightness']:.1f}", callback_data='ascii_brightness_info'),
            InlineKeyboardButton("Brightness -", callback_data='ascii_brightness_down')
        ],
        [
            InlineKeyboardButton("Width +", callback_data='ascii_width_up'),
            InlineKeyboardButton(f"{settings['width']}", callback_data='ascii_width_info'),
            InlineKeyboardButton("Width -", callback_data='ascii_width_down')
        ],
        [
            InlineKeyboardButton("Color Mode: " + settings['color_mode'].upper(), callback_data='ascii_cycle_color')
        ],
        [
            InlineKeyboardButton("True Color: " + ("ON" if converter.true_color else "OFF"), callback_data='ascii_toggle_true_color'),
            InlineKeyboardButton("Invert: " + ("ON" if converter.invert else "OFF"), callback_data='ascii_toggle_invert')
        ],
        [
            InlineKeyboardButton("Reset Settings", callback_data='ascii_reset')
        ],
        [
            InlineKeyboardButton("🏠 Main Menu", callback_data='return_main')
        ]
    ]
    
    ascii_message = (
        "🎨 ASCII Art Converter Settings 🎨\n\n"
        "Current Settings:\n"
        f"• Contrast: {settings['contrast']:.1f}\n"
        f"• Brightness: {settings['brightness']:.1f}\n"
        f"• Width: {settings['width']}\n"
        f"• Color Mode: {settings['color_mode'].upper()}\n"
        f"• True Color: {'ON' if converter.true_color else 'OFF'}\n"
        f"• Invert: {'ON' if converter.invert else 'OFF'}\n\n"
        "Send me any image to convert it with these settings! 📸\n"
        "(This menu will auto-delete in 60 seconds if not used)"
    )
    
    if update.callback_query:
        sent_message = await update.callback_query.edit_message_text(
            ascii_message,
            reply_markup=InlineKeyboardMarkup(ascii_menu_keyboard)
        )
    else:
        sent_message = await update.message.reply_text(
            ascii_message,
            reply_markup=InlineKeyboardMarkup(ascii_menu_keyboard)
        )
    
    # Initialize or update last interaction time
    context.user_data['last_ascii_interaction'] = datetime.now().timestamp()
    
    # Create new deletion task
    async def delete_menu():
        try:
            current_time = datetime.now().timestamp()
            last_interaction = context.user_data.get('last_ascii_interaction', 0)
            
            # Only delete if no interaction in the last 60 seconds
            if current_time - last_interaction >= 60:
                await sent_message.delete()
        except Exception:
            pass  # Message might already be deleted
    
    # Schedule the deletion task
    context.user_data['menu_deletion_task'] = asyncio.create_task(
        asyncio.sleep(60)
    )
    context.user_data['menu_deletion_task'].add_done_callback(
        lambda _: asyncio.create_task(delete_menu())
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos and convert them to ASCII art."""
    # Check if this is a reply to the ASCII art prompt
    if not update.message.reply_to_message or \
       update.message.reply_to_message.from_user.id != context.bot.id or \
       not "Send me any image to convert it with these settings!" in update.message.reply_to_message.text:
        return

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
            "Converting to ASCII art with enhanced contrast..."
        )

        # Download the photo
        await photo.download_to_drive(photo_path)

        if not os.path.exists(photo_path):
            raise FileNotFoundError("Failed to download the photo")

        try:
            # Convert to ASCII art with maximum supported settings
            if 'ascii_settings' not in context.user_data:
                context.user_data['ascii_settings'] = {
                    'contrast': 1.0,
                    'brightness': 1.0,
                    'width': 250
                }
            
            settings = context.user_data['ascii_settings']
            base_width = settings['width']
            
            # Calculate width to maintain aspect ratio
            with Image.open(photo_path) as img:
                aspect_ratio = img.height / img.width
                # If image is taller than wide, adjust width to maintain proportions
                if aspect_ratio > 1:
                    width = int(base_width / (aspect_ratio * 0.5))  # Compensate for ASCII char height/width ratio
                else:
                    width = base_width
            
            # Update converter settings
            converter.contrast = settings['contrast']
            converter.brightness = settings['brightness']
            
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

            # Send the ASCII art image (don't auto-delete this)
            with open(image_path, 'rb') as image_file:
                await update.message.reply_photo(
                    photo=image_file,
                    caption="🎨 Here's your spiritual ASCII art! ✨"
                )

            # Cancel any existing menu deletion task
            if 'menu_deletion_task' in context.user_data:
                context.user_data['menu_deletion_task'].cancel()

            # Show ASCII menu again with auto-delete
            await handle_ascii_menu(update, context)

            # Reset the last interaction time
            context.user_data['last_ascii_interaction'] = datetime.now().timestamp()

        except Exception as e:
            logger.error(f"Error during ASCII conversion: {str(e)}")
            raise Exception(f"Failed to convert image to ASCII: {str(e)}")

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
    token = "7609264045:AAFMBSKKNAayPuiFnDhu4WCL3AbpG8a24ZI"
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("tree", tree))  # Changed from start to tree
    application.add_handler(CommandHandler("bday", bday))
    application.add_handler(CommandHandler("vibe", vibe))
    application.add_handler(CommandHandler("num", num))
    application.add_handler(CallbackQueryHandler(menu_handler))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_messages))

    # Start the bot
    print("Sacred Tree of Life Bot started! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 