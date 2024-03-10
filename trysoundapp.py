import streamlit as st
import speech_recognition as sr
import trysoundlib as trysound
# Streamlit UI
st.title("Speech to Text")
textbox = st.text_area("Text Box", height=250)
mic_button = st.button("🎙️ Speak")

if mic_button:
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=1)  # Ensure correct microphone device index
    transcription = trysound.recognize_speech_from_mic(recognizer, microphone)
    st.write("You said: ", transcription)
    st.text_area("Text Box", value=transcription, height=250)
