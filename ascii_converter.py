import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import os
from typing import Tuple, Optional, List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class ASCIIArtConverter:
    """ASCII art converter based on GeeksForGeeks methodology."""
    
    # Extended ASCII characters ordered by increasing density with density values
    ASCII_CHARS_WITH_DENSITY = [
        (" ", 0.0),      # Level 0 (empty space)
        (".", 0.15),     # Level 1 (light dot)
        ("'", 0.2),      # Level 2 (apostrophe)
        ("`", 0.2),      # Level 3 (backtick)
        ("^", 0.25),     # Level 4 (caret)
        ("\"", 0.25),    # Level 5 (double quote)
        (",", 0.3),      # Level 6 (comma)
        (":", 0.35),     # Level 7 (colon)
        (";", 0.35),     # Level 8 (semicolon)
        ("I", 0.4),      # Level 9 (capital i)
        ("l", 0.4),      # Level 10 (lowercase L)
        ("!", 0.45),     # Level 11 (exclamation)
        ("i", 0.45),     # Level 12 (lowercase i)
        (">", 0.45),     # Level 13 (greater than)
        ("<", 0.45),     # Level 14 (less than)
        ("~", 0.5),      # Level 15 (tilde)
        ("+", 0.5),      # Level 16 (plus)
        ("_", 0.55),     # Level 17 (underscore)
        ("-", 0.55),     # Level 18 (minus)
        ("?", 0.6),      # Level 19 (question mark)
        ("]", 0.6),      # Level 20 (right bracket)
        ("[", 0.6),      # Level 21 (left bracket)
        ("}", 0.6),      # Level 22 (right brace)
        ("{", 0.6),      # Level 23 (left brace)
        ("1", 0.65),     # Level 24 (one)
        (")", 0.65),     # Level 25 (right parenthesis)
        ("(", 0.65),     # Level 26 (left parenthesis)
        ("|", 0.7),      # Level 27 (vertical bar)
        ("\\", 0.7),     # Level 28 (backslash)
        ("/", 0.7),      # Level 29 (forward slash)
        ("t", 0.75),     # Level 30 (lowercase t)
        ("f", 0.75),     # Level 31 (lowercase f)
        ("j", 0.75),     # Level 32 (lowercase j)
        ("r", 0.75),     # Level 33 (lowercase r)
        ("x", 0.8),      # Level 34 (lowercase x)
        ("n", 0.8),      # Level 35 (lowercase n)
        ("u", 0.8),      # Level 36 (lowercase u)
        ("v", 0.8),      # Level 37 (lowercase v)
        ("c", 0.85),     # Level 38 (lowercase c)
        ("z", 0.85),     # Level 39 (lowercase z)
        ("X", 0.85),     # Level 40 (uppercase X)
        ("Y", 0.85),     # Level 41 (uppercase Y)
        ("U", 0.9),      # Level 42 (uppercase U)
        ("J", 0.9),      # Level 43 (uppercase J)
        ("C", 0.9),      # Level 44 (uppercase C)
        ("L", 0.9),      # Level 45 (uppercase L)
        ("Q", 0.9),      # Level 46 (uppercase Q)
        ("0", 0.95),     # Level 47 (zero)
        ("O", 0.95),     # Level 48 (uppercase O)
        ("Z", 0.95),     # Level 49 (uppercase Z)
        ("*", 0.95),     # Level 50 (asterisk)
        ("#", 0.95),     # Level 51 (hash)
        ("M", 1.0),      # Level 52 (uppercase M)
        ("W", 1.0),      # Level 53 (uppercase W)
        ("&", 1.0),      # Level 54 (ampersand)
        ("8", 1.0),      # Level 55 (eight)
        ("%", 1.0),      # Level 56 (percent)
        ("B", 1.0),      # Level 57 (uppercase B)
        ("@", 1.0),      # Level 58 (at sign)
        ("$", 1.0)       # Level 59 (dollar sign)
    ]
    
    ASCII_CHARS = [char for char, _ in ASCII_CHARS_WITH_DENSITY]
    ASCII_DENSITIES = [density for _, density in ASCII_CHARS_WITH_DENSITY]

    def __init__(self, contrast: float = 1.5, brightness: float = 1.2,
                 invert: bool = False, color_mode: str = 'none', true_color: bool = False,
                 num_threads: int = None, chunk_size: int = 50,
                 use_multiprocessing: bool = False, use_colored_chars: bool = False):
        """Initialize the ASCII art converter with enhanced performance settings."""
        self.contrast = contrast
        self.brightness = brightness
        self.invert = invert
        self.color_mode = color_mode
        self.true_color = true_color
        self.num_threads = num_threads or max(1, multiprocessing.cpu_count())
        self.chunk_size = chunk_size
        self.use_multiprocessing = use_multiprocessing
        self.use_colored_chars = use_colored_chars  # New parameter for colored ASCII chars
        self._initialize_color_map()
        
        # Create cache dictionaries
        self._color_cache = {}
        self._luminance_cache = {}
        self._ascii_cache = {}

    def _get_cached_value(self, cache_dict, key, compute_func):
        """Get a cached value or compute and cache it."""
        if key not in cache_dict:
            cache_dict[key] = compute_func()
            if len(cache_dict) > 1024:  # Limit cache size
                cache_dict.pop(next(iter(cache_dict)))
        return cache_dict[key]

    def calculate_luminance(self, color: tuple) -> float:
        """Calculate luminance with manual caching."""
        return self._get_cached_value(
            self._luminance_cache,
            color,
            lambda: (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255.0
        )

    def process_color(self, color: tuple, intensity: float, char_density: float) -> tuple:
        """Optimized color processing with manual caching."""
        cache_key = (color, intensity, char_density)
        return self._get_cached_value(
            self._color_cache,
            cache_key,
            lambda: self._process_color_impl(color, intensity, char_density)
        )

    def _process_color_impl(self, color: tuple, intensity: float, char_density: float) -> tuple:
        """Internal color processing implementation."""
        if not self.true_color:
            base_color = self.color_map[self.color_mode]
            return tuple(int(c * intensity) for c in base_color)
        
        r, g, b = color
        luminance = self.calculate_luminance(color)
        
        # Optimized color preservation logic
        if char_density < 0.1:
            brightness = 255.0
            density_factor = 0.95
        elif char_density < 0.3:
            brightness = 230.0
            density_factor = 0.85
        elif char_density < 0.6:
            brightness = 200.0 * (1.0 - (char_density - 0.3) / 0.3)
            density_factor = 0.7
        else:
            brightness = 150.0 * (1.0 - (char_density - 0.6) / 0.4)
            density_factor = 0.5
        
        # Optimized color calculations using numpy
        colors = np.array([r, g, b], dtype=np.float32)
        
        # Apply gamma correction
        gamma = 1.0 + (0.2 * (1.0 - luminance))
        colors = np.power(colors / 255.0, 1.0/gamma) * 255.0
        
        # Enhance saturation
        avg = np.mean(colors)
        colors = avg + (colors - avg) * 1.2
        
        # Calculate final colors
        intensity_factor = 0.7 + (0.3 * intensity)
        colors = brightness * (colors/255.0) * density_factor * intensity_factor
        
        # Clip values and convert to integers
        return tuple(np.clip(colors, 0, 255).astype(np.uint8))

    def pixel_to_ascii(self, pixel_value: int) -> str:
        """Convert pixel intensity to ASCII character with manual caching."""
        return self._get_cached_value(
            self._ascii_cache,
            pixel_value,
            lambda: self._pixel_to_ascii_impl(pixel_value)
        )

    def _pixel_to_ascii_impl(self, pixel_value: int) -> str:
        """Internal pixel to ASCII conversion implementation."""
        char_index = int((pixel_value / 255) * (len(self.ASCII_CHARS) - 1))
        return self.ASCII_CHARS[char_index]

    def _initialize_color_map(self):
        """Initialize color mapping for different modes."""
        self.color_map = {
            "none": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128)
        }
    
    def resize_image(self, image: Image.Image, new_width: int) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        aspect_ratio = image.height / image.width
        # Don't compensate for character aspect ratio anymore, preserve original ratio
        new_height = int(aspect_ratio * new_width)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def process_image_chunk(self, args: tuple) -> Tuple[str, List[List[str]], List[List[tuple]]]:
        """Process image chunk with numpy optimizations."""
        pixels, original, start_row, end_row = args
        
        # Pre-allocate arrays for better performance
        chunk_height = end_row - start_row
        chunk_width = pixels.shape[1]
        
        ascii_chars = np.empty((chunk_height, chunk_width), dtype='U1')
        ascii_colors = np.empty((chunk_height, chunk_width, 3), dtype=np.uint8)
        
        # Process entire chunk using numpy operations
        normalized_pixels = (pixels[start_row:end_row] / 255.0 * (len(self.ASCII_CHARS) - 1)).astype(int)
        ascii_chars = np.array(self.ASCII_CHARS)[normalized_pixels]
        
        # Process colors
        ascii_colors = original[start_row:end_row, :, :3].copy()
        
        # Convert to Python lists and strings
        ascii_str = '\n'.join(''.join(row) for row in ascii_chars)
        ascii_chars_list = ascii_chars.tolist()
        ascii_colors_list = [
            [tuple(color) for color in row]
            for row in ascii_colors
        ]
        
        return ascii_str, ascii_chars_list, ascii_colors_list
    
    def draw_character_block(self, draw: ImageDraw.Draw, x: int, y: int, 
                           char: str, color: tuple, font: ImageFont.ImageFont,
                           char_width: int, char_height: int) -> None:
        """Draw a character block with enhanced color processing."""
        # Get character density
        char_index = self.ASCII_CHARS.index(char)
        char_density = self.ASCII_DENSITIES[char_index]
        
        if self.use_colored_chars:
            # For colored chars mode, use white/black background
            bg_color = (0, 0, 0) if self.invert else (255, 255, 255)
            draw.rectangle([x, y, x + char_width, y + char_height], fill=bg_color)
            
            if self.true_color:
                # Get original color components
                r, g, b = color
                
                # Apply density-based intensity
                intensity = max(0.4, char_density)  # Minimum intensity of 0.4
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)
                
                char_color = (r, g, b)
            else:
                base_color = self.color_map[self.color_mode]
                # For solid colors, use higher minimum intensity
                intensity = max(0.6, char_density)
                char_color = tuple(int(c * intensity) for c in base_color)
            
            if self.invert:
                char_color = tuple(255 - c for c in char_color)
            
            # Draw the character in the processed color
            draw.text((x, y), char, font=font, fill=char_color)
        else:
            # Original tile mode
            # Calculate character intensity with better precision
            intensity = char_density
            # Process color with enhanced algorithms
            final_color = self.process_color(color, intensity, char_density)
            if self.invert:
                final_color = tuple(255 - c for c in final_color)
                
            draw.rectangle([x, y, x + char_width, y + char_height], fill=final_color)
            # Calculate optimal text color for contrast
            luminance = (0.299 * final_color[0] + 0.587 * final_color[1] + 0.114 * final_color[2]) / 255.0
            text_color = (0, 0, 0) if luminance > 0.6 else (255, 255, 255)
            # Add slight offset for better readability
            offset = 1 if char_density > 0.5 else 0
            draw.text((x + offset, y), char, font=font, fill=text_color)
    
    def image_to_ascii(self, image_path: str, width: int = 200,
                      save_text: bool = True, save_image: bool = True,
                      output_dir: str = "ascii_output",
                      output_image_path: Optional[str] = None) -> Tuple[str, Optional[Image.Image]]:
        """Convert image to ASCII art with enhanced parallel processing."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with Image.open(image_path) as image:
                # Optimize image preprocessing
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Increased max_width to 500 for better detail with 200 default width
                max_width = min(width, 500)
                image = self.resize_image(image, max_width)
                
                # Enhanced color processing for colored chars mode
                if self.use_colored_chars:
                    # Enhance color saturation
                    image = ImageEnhance.Color(image).enhance(2.0)
                    # Enhance contrast
                    image = ImageEnhance.Contrast(image).enhance(self.contrast * 1.5)
                    # Enhance brightness
                    image = ImageEnhance.Brightness(image).enhance(self.brightness * 1.2)
                else:
                    # Original enhancement for tile mode
                    image = ImageEnhance.Color(image).enhance(1.2)
                
                # Convert to numpy arrays with optimal dtypes
                grayscale = np.array(image.convert("L"), dtype=np.uint8)
                original = np.array(image, dtype=np.uint8)
                
                # Apply enhancements using numpy operations
                if not self.use_colored_chars:
                    # Only apply to grayscale in tile mode
                    grayscale = np.clip(grayscale * self.contrast, 0, 255).astype(np.uint8)
                    grayscale = np.clip(grayscale * self.brightness, 0, 255).astype(np.uint8)
                
                if self.invert:
                    grayscale = 255 - grayscale
                
                # Process chunks in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    height = grayscale.shape[0]
                    chunks = [
                        (grayscale, original, start, min(start + self.chunk_size, height))
                        for start in range(0, height, self.chunk_size)
                    ]
                    
                    results = list(executor.map(self.process_image_chunk, chunks))
                
                # Combine results
                ascii_str = ""
                ascii_chars = []
                ascii_colors = []
                
                for chunk_str, chunk_chars, chunk_colors in results:
                    ascii_str += chunk_str
                    ascii_chars.extend(chunk_chars)
                    ascii_colors.extend(chunk_colors)
                
                # Save outputs
                if save_text:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    text_path = os.path.join(output_dir, f"{base_name}_ascii.txt")
                    with open(text_path, "w") as f:
                        f.write(ascii_str)
                
                if save_image:
                    ascii_image = self._create_ascii_image(
                        ascii_chars, ascii_colors, output_dir, output_image_path, image_path
                    )
                    return ascii_str, ascii_image
                
                return ascii_str, None
                
        except Exception as e:
            logger.error(f"Error in image_to_ascii: {str(e)}")
            raise

    def _create_ascii_image(self, ascii_chars, ascii_colors, output_dir, output_image_path, image_path):
        """Create ASCII image with optimized processing."""
        try:
            font = ImageFont.truetype("Courier New", 10)
        except:
            font = ImageFont.load_default()
        
        # Get original image dimensions to calculate proper scaling
        with Image.open(image_path) as original_img:
            orig_aspect = original_img.height / original_img.width
        
        # Calculate dimensions that preserve aspect ratio
        char_width = font.getbbox("A")[2]
        char_height = font.getbbox("A")[3]
        
        # Calculate image width based on number of characters
        img_width = char_width * len(ascii_chars[0])
        # Calculate height to match original aspect ratio
        img_height = int(img_width * orig_aspect)
        # Adjust character height to fit the desired image height
        char_height = img_height // len(ascii_chars)
        
        bg_color = "black" if self.invert else "white"
        ascii_image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(ascii_image)
        
        # Process image in parallel chunks
        def process_image_row(row_data):
            i, (line_chars, line_colors) = row_data
            y = i * char_height
            for j, (char, color) in enumerate(zip(line_chars, line_colors)):
                x = j * char_width
                self.draw_character_block(draw, x, y, char, color, font, char_width, char_height)
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            row_data = enumerate(zip(ascii_chars, ascii_colors))
            list(executor.map(process_image_row, row_data))
        
        # Save with optimal settings
        save_kwargs = {
            'optimize': True,
            'quality': 95
        }
        
        if output_image_path:
            ascii_image.save(output_image_path, **save_kwargs)
        else:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_ascii.png")
            ascii_image.save(output_path, **save_kwargs)
        
        return ascii_image

if __name__ == "__main__":
    # Example usage with maximum performance settings
    converter = ASCIIArtConverter(
        contrast=1.5,
        brightness=1.2,
        invert=False,
        color_mode='none',
        true_color=True,
        num_threads=multiprocessing.cpu_count(),  # Use all CPU cores
        chunk_size=25,
        use_multiprocessing=True,  # Enable multiprocessing for maximum performance
        use_colored_chars=True  # Enable colored characters
    )
    
    # Process the most recent image
    generated_dir = "generated_images"
    if os.path.exists(generated_dir):
        images = [f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            most_recent = max(images, key=lambda x: os.path.getctime(os.path.join(generated_dir, x)))
            input_image = os.path.join(generated_dir, most_recent)
            
            print(f"Converting most recent image: {input_image}")
            try:
                ascii_str, ascii_img = converter.image_to_ascii(
                    input_image,
                    width=200,  # Increased width for better detail
                    save_text=True,
                    save_image=True,
                    output_dir="ascii_output"
                )
                print("\nASCII Art Preview:")
                print(ascii_str)
            except Exception as e:
                print(f"Error converting image: {str(e)}") 