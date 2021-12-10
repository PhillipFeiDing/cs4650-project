import string
import re

from .abbreviations import abbreviations

# Replace all punctuations
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

# Remove all URLs, replace by URL
def remove_URL(text):
    return re.compile(r'https?://\S+|www\.\S+').sub(r'URL',text)

# Remove HTML beacon
def remove_HTML(text):
    return re.compile(r'<.*?>').sub(r'',text)

# Remove non printable characters
def remove_not_ASCII(text):
    return ''.join([word for word in text if word in string.printable])

# Replace all abbreviations
def replace_abbrev(text):
    # Change an abbreviation by its true meaning
    def word_abbrev(word):
        return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
    string = ""
    for word in text.split():
        string += word_abbrev(word) + " "        
    return string

# Remove @ and mention, replace by USER
def remove_mention(text):
    return re.compile(r'@\S+').sub(r'USER',text)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    return re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*').sub(r'NUMBER', text)

# Remove all emojis, replace by EMOJI
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text)

# Replace some others smileys with SADFACE
def transcription_sad(text):
    return re.compile(r'[8:=;][\'\-]?[(\\/]').sub(r'SADFACE', text)

# Replace some smileys with SMILE
def transcription_smile(text):
    return re.compile(r'[8:=;][\'\-]?[)dDp]').sub(r'SMILE', text)

# Replace <3 with HEART
def transcription_heart(text):
    heart = re.compile(r'<3')
    return heart.sub(r'HEART', text)

def clean_text(text):
    # remove non text
    text = remove_URL(text)
    text = remove_HTML(text)
    text = remove_not_ASCII(text)
    
    # replace abbreviations, @ and number
    text = replace_abbrev(text)  
    text = remove_mention(text)
    text = remove_number(text)
    
    # remove emojis / smileys
    text = remove_emoji(text)
    text = transcription_sad(text)
    text = transcription_smile(text)
    text = transcription_heart(text)
    
    # remove punctuations
    text = remove_punctuation(text)
  
    return text