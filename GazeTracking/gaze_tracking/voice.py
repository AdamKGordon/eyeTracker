#pip install pynput
#pip install SpeechRecognition==2.0.1

from pynput.mouse import Button, Controller
import speech_recognition as sr
mouse = Controller()
r = sr.Recognizer()
with sr.Microphone() as source:                # use the default microphone as the audio source
    r.adjust_for_ambient_noise(source)         # to reduce noise 
    audio = r.listen(source)                   # listen for the first phrase and extract it into audio data

try:
    print("You said " + r.recognize(audio))    # recognize speech using Google Speech Recognition
    if (r.recognize(audio) == 'hey iclicker left click' or r.recognize(audio) == 'left click'):
        mouse.click(Button.left,2)
    
    elif (r.recognize(audio) == 'hey iclicker right click' or r.recognize(audio) == 'right click'):
        mouse.click(Button.right,1)

except LookupError:                            # speech is unintelligible
    print("Could not understand audio")

