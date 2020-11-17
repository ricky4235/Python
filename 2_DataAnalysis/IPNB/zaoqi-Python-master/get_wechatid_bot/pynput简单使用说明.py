from pynput.mouse import Button, Controller
import time 

mouse = Controller()
print(mouse.position)
time.sleep(3)
print('鼠标现在的位置在 {0}'.format(mouse.position))
print(mouse.position)


#点击
mouse.press(Button.left)
mouse.release(Button.left)

#双击
mouse.click(Button.left, 1)