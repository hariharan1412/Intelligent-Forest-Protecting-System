from cProfile import label
from tkinter import *
from PIL import Image, ImageTk
import threading
from teamstartup_fire_detection import fire_detection
from teamstartup_chainsaw_detect import chainsaw_detect


fd = fire_detection()
cd = chainsaw_detect()

def activate(root):
    root.destroy()
    t1 = threading.Thread(target=fd.main)
    t2 = threading.Thread(target=cd.detect)
    t1.start()
    t2.start()

def main():

    root = Tk()

    root.title("GOD SENSE")

    FILENAME = "Untitled presentation.jpg"

    canvas = Canvas(root, width=950, height=570)

    canvas.pack()

    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(470, 300, image=tk_img)


    start_btn = Button(root, text = "  Walk In ", command=lambda:activate(root), anchor = 'w',
                        width = 10, activebackground = "#33B5E5")

    start_button_window = canvas.create_window(440, 235, anchor='nw', window=start_btn)    

    root.mainloop()

if __name__ == "__main__":
    main()