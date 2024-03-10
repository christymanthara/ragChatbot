import streamlit as st
import speech_recognition as sr

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        st.write("Say something!")
        audio = recognizer.listen(source)
    
    # Recognize speech using Google Web Speech API
    try:
        transcription = recognizer.recognize_google(audio, language='en-US')
        return transcription
    except sr.RequestError:
        # API was unreachable or unresponsive
        return "API unavailable"
    except sr.UnknownValueError:
        # Speech was unintelligible
        return "Unable to recognize speech. Can you repeat that in English"