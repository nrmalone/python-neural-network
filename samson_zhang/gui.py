# https://www.youtube.com/watch?v=4ehHuDDH-uc&ab_channel=pycad

from tkinter import *
from PIL import Image, ImageTk, ImageGrab


global progress
progress = 0

# GUI setup
app = Tk()
app.geometry("720x600")
app.title("Digit Recognition")
app.configure(bg='black')
app.resizable(False, False)


# Helper functions and button callbacks
def get_pos(event):
    global x, y
    x, y = event.x, event.y

def draw(event):
    global x, y
    canvas.create_line(x, y, event.x, event.y, fill='white', width=10)
    x, y = event.x, event.y

def submit():
    pass

def clear():
    canvas.delete("all")


# GUI elements
instructionLabel = Label(text="Draw a digit using your mouse cursor. Press 'Submit' to have the AI predict the digit or 'Clear' to clear the canvas.", bg='black', fg='white')
instructionLabel.place(x=0, y=0)

canvas = Canvas(app, bg='black')
canvas.configure(width=528, height=528, highlightthickness=2, highlightcolor='white')
canvas.place(x=0, y=30)

canvas.bind("<Button-1>", get_pos)
canvas.bind("<B1-Motion>", draw)

submitBtn = Button(app, text='Submit', command=submit, width=20, bg='black', fg='white')
submitBtn.place(x=538, y=30)

clearBtn = Button(app, text='Clear', command=clear, width=20, bg='black', fg='white')
clearBtn.place(x=538, y=80)

if __name__ == '__main__':
    app.mainloop()