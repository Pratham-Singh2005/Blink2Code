# morse_translator.py - Morse Code Translator

MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--', 
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
    '8': '---..', '9': '----.', '0': '-----', ' ': '/'
}

# Reverse dictionary for decoding
REVERSE_MORSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}


def blinks_to_morse(blink_sequence):
    """ Converts a sequence of short ('S') and long ('L') blinks into Morse code. """
    return ''.join(['.' if blink == 'S' else '-' for blink in blink_sequence])


def morse_to_text(morse_code):
    """ Converts Morse code to text. """
    words = morse_code.split(' / ')  # Split words
    translated_words = []

    for word in words:
        letters = word.split(' ')  # Split letters
        translated_word = ''.join(REVERSE_MORSE_DICT.get(letter, '') for letter in letters)
        translated_words.append(translated_word)

    return ' '.join(translated_words)
