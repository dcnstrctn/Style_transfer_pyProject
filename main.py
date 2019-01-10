# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:28:36 2018

@author: XZ
"""
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from style_transfer import *
root = Tk()
root.title("Style Transfer")

screenwidth = root.winfo_screenwidth()  
screenheight = root.winfo_screenheight()

frmleft=Frame(root)
frmleft.pack(side='left')
frmright=Frame(root)
frmright.pack(side='right')
frmtop=Frame(frmleft)
frmtop.pack(side='top')
frmbottom=Frame(frmleft)
frmbottom.pack(side='bottom')

content_path="styles/pepe.jpg"
style_path="styles/pepe.jpg"


def getPath():
    return filedialog.askopenfilename(parent=root, title='Choose an image.')

#function to be called when mouse is clicked
def printcoords(target, File):
    im = PIL.Image.open(File)
    (x,y)=im.size
    im=im.resize((300,300), PIL.Image.ANTIALIAS)
    filename = ImageTk.PhotoImage(im)
    target.image = filename  # <--- keep reference of your image
    target.create_image(0,0,anchor='nw',image=filename)
    
def contentClk(target):
    global content_path
    content_path=getPath()
    printcoords(target, content_path)
    
def styleClk(target):
    global style_path
    style_path=getPath()
    printcoords(target, style_path)

canvas1 = Canvas(frmtop, width = 300, height = 300, bg="white")
canvas1.pack()
canvas2 = Canvas(frmbottom, width = 300, height = 300, bg="white")
canvas2.pack()
canvas3 = Canvas(frmright,width = 500, height = 500, bg="white")
canvas3.pack()
label=Label(frmright, text="点此转换")
label.pack()

def transferClk(content_path, style_path, **params):
    global label
    label["text"]="正在转换中……请耐心等待"
    label.pack()
    img = style_transfer(content_path, style_path, **params)
    label["text"]="转换成功！"
    label.pack()
    global image 
    image = img
    #display image
    img=img.resize((500,500), PIL.Image.ANTIALIAS)
    filename = ImageTk.PhotoImage(img)
    global canvas3
    canvas3.image = filename  # <--- keep reference of your image
    canvas3.create_image(0,0,anchor='nw',image=filename)
    
def saveClk():
    image.save(filedialog.asksaveasfilename())

Button(frmtop,text='选择原图',command=lambda: contentClk(canvas1)).pack()
Button(frmbottom,text='选择风格图片',command=lambda: styleClk(canvas2)).pack()

params = {
    'image_size' : 224,
    'style_size' : 224,
    'content_layer' : 3,
    'content_weight' : 3e-2, 
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (200000, 800, 12, 1),
    'tv_weight' : 2e-2,
}

Button(frmright,text='转换！',command=lambda: transferClk(content_path, style_path, **params)).pack()
Button(frmright,text='另存为',command=lambda: saveClk()).pack()

root.mainloop()


