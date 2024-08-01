from tkinter import *
from tkinter import ttk
import threading
import time
import rnn
import gui

loadingScreen = Tk()
loadingScreen.title("Loading...")
loadingScreen.geometry("250x50")

progressBar = ttk.Progressbar(loadingScreen, orient="horizontal", length=100, mode="determinate", maximum=100)
progressBar.place(x=10, y=10, width=230)

def update_progress():
    threading.Thread(target=rnn.train).start()
    for x in range(int(100/rnn.iterations)):
        progressBar['value']+=1
        loadingScreen.update()
        time.sleep(1)
        

def main():
    loadingScreen.mainloop()

    threading.Thread(target=update_progress).start()

    if rnn.completed:
        loadingScreen.destroy()
        gui.app.mainloop()    

if __name__ == "__main__":
    main()