import pyautogui
import time


time.sleep(5)
count=0

#Inside Games_Img folder, you adjust on what Game folder you want to put the
#images at.
GAME_PATH = '/Game9/'

while True:
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'Games_Img'+GAME_PATH+'screenshot'+str(count)+'.png')
    count+=1
    time.sleep(12)