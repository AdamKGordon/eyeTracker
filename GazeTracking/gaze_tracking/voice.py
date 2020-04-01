#pip install pynput
#pip install SpeechRecognition==2.0.1
#from pynput.mouse import Button, Controller
import speech_recognition as sr

try:
    from .mouse import Mouse
except:
    from mouse import Mouse


def debug_print(str):
    if False:
        print(str)

def voice():
    #mouse = Controller()
    mouse = Mouse()
    r = sr.Recognizer()

    while True:
        debug_print("1")
        with sr.Microphone() as source:                # use the default microphone as the audio source
            debug_print("2")
            r.adjust_for_ambient_noise(source)         # to reduce noise 
            debug_print("3")
            audio = r.listen(source)                   # listen for the first phrase and extract it into audio data
            debug_print("4")

        try:
            print("You said " + r.recognize(audio))    # recognize speech using Google Speech Recognition
            if (r.recognize(audio) == 'hey iclicker left click' or r.recognize(audio) == 'left click'):
                mouse.left_click()
    
            elif (r.recognize(audio) == 'hey iclicker right click' or r.recognize(audio) == 'right click'):
                mouse.right_click()

        except LookupError:                            # speech is unintelligible
            print("Could not understand audio")
            print("left clicking...")
            debug_print("5")
            debug_print("6")
