import speech_recognition as sr # type: ignore

def get_voice_answer() -> str:
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak your answer...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError:
        return "Speech service unavailable"
