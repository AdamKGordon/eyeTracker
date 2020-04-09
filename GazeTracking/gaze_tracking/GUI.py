import os
from tkinter import *


class App:
    def __init__(self, root):
        # Add two frames
        frame1 = Frame(root)
        frame1.pack(padx=20, pady=20)
        frame2 = Frame(root)
        frame2.pack(padx=20, pady=20)
        # Add background image(logo)
        self.logo = PhotoImage(file="./static/logo.png")
        self.label1 = Label(frame1, justify="left", compound=CENTER, fg="blue", image=self.logo,
                            font=("Times New Roman", 50))
        self.label1.pack(side=LEFT)
        self.initiate = Button(frame2, text="Initiate Eye-Clicker", fg="blue", bg="red", width=20, height=10,
                               command=self.initialization)
        self.initiate.pack(padx=200, pady=20, side=LEFT)
        self.calibration = Button(frame2, text="Start calibration", fg="blue", bg="red", width=20, height=10)
        self.calibration.pack(padx=200, pady=20, side=RIGHT)

    def initialization(self):
        os.system("python ../example.py")


root = Tk()
root.title("Eye-Clicker")

app = App(root)

root.mainloop()
