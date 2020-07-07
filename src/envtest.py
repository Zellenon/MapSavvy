import threading
import numpy as np
import numpy.random as nprand
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty
from kivy.lang.builder import Builder
from kivy.uix.button import Button

print('a')

def timedFunction():
    a = nprand.randint(-10,10,2000)
    b = np.zeros((2000,2000))
    for i in range(0,len(a)):
        for j in range(0,len(a)):
            b[i][j] = a[i] * a[j]
        if (i % 100 == 0):
            print (i)
    print(len(a))

Builder.load_string('''
[SideBar@BoxLayout]:
    content: content
    orientation: 'vertical'
    size_hint: .2,1
    BoxLayout:
        orientation: 'vertical'
        # just add a id that can be accessed later on
        id: content

<Root>:
    Button:
        center_x: root.center_x
        text: 'press to add_widgets'
        size_hint: .2, .2
        on_press:
            sb.content.clear_widgets()
            root.load_content(sb.content, canvas)
    SideBar:
        id: sb
    Widget:
        size_hint: 1, 1
        id: canvas
''')

class Root(BoxLayout):

    def load_content(self, content, canvas):
        threading.Thread(target=timedFunction, daemon=True).start()
        for but in range(20):
            content.add_widget(Button(text=str(but)))
            canvas.Rectangle(pos=(10,10), size=(10,10))
        


class MyApp(App):
    def build(self):
        return Root()

if __name__ == "__main__":
    app = MyApp()
    app.run()