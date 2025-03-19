import speech_recognition as sr
import string

def listen_to_paragraph(paragraph):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please read the following paragraph:")
        print(paragraph)
        print("Listening...")
        
        # Adjust for ambient noise and record audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Use Google Web Speech API to recognize the audio
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def check_for_dyslexia(original, detected):
    # Normalize text: lowercase and remove punctuation
    original_words = original.lower().translate(str.maketrans('', '', string.punctuation)).split()
    detected_words = detected.lower().translate(str.maketrans('', '', string.punctuation)).split()
    
    # Find missing words
    missing_words = [word for word in original_words if word not in detected_words]
    missing_count = len(missing_words)
    
    print(f"\nMissing words: {missing_words}")
    print(f"Number of missing words: {missing_count}")
    
    # Check if 3 or more words are missing
    if missing_count >= 2:
        print("You may have dyslexia. Please consider consulting a professional.")
    else:
        print("You read the paragraph well!")

# Define the paragraph to be read
paragraph = """Artificial intelligence (AI) is intelligence demonstrated by machines"""

# Call the function to listen to the paragraph
detected_text = listen_to_paragraph(paragraph)

# Optionally, you can compare the detected text with the original paragraph
if detected_text:
    print("\nComparison with the original paragraph:")
    print("Original:", paragraph)
    print("Detected:", detected_text)
    
    # Check for dyslexia indicators
    check_for_dyslexia(paragraph, detected_text)