import speech_recognition as sr
from pynput.mouse import Button, Controller

mouse = Controller()
r = sr.Recognizer()
with sr.Microphone() as source:  # use the default microphone as the audio source
    audio = r.listen(source)  # listen for the first phrase and extract it into audio data

try:
    print("You said " + r.recognize(audio))  # recognize speech using Google Speech Recognition
    if (r.recognize(audio) == 'hey iclicker left click'):
        mouse.click(Button.left, 2)

    elif (r.recognize(audio) == 'hey iclicker right click'):
        mouse.click(Button.right, 1)

except LookupError:  # speech is unintelligible
    print("Could not understand audio")
