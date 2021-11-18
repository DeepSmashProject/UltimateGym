import os
from PIL import Image
from numpy.core.fromnumeric import argmax
import pyocr
import sys
import cv2
import numpy as np
def ditect():
    #インストールしたTesseract-OCRのパスを環境変数「PATH」へ追記する。
    #OS自体に設定してあれば以下の2行は不要
    #path='C:\\Program Files\\Tesseract-OCR'
    TESSERACT_PATH = '/usr/local/share/tessdata'
    TESSDATA_PATH = '/usr/local/share/tessdata'

    os.environ["PATH"] += os.pathsep + TESSERACT_PATH
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()), tools)
    # Ex: Will use tool 'libtesseract'
    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = langs[3]
    print("Will use lang '%s'" % (lang))

    #import sys
    #sys.path.append('/usr/local/share/tessdata/eng.traineddata')
    #pyocrへ利用するOCRエンジンをTesseractに指定する。
    
    #OCR対象の画像ファイルを読み込む
    img = Image.open("./test.png")
    
    #画像から文字を読み込む
    builder = pyocr.builders.TextBuilder(tesseract_layout=6)
    text = tool.image_to_string(img, lang=lang, builder=builder)
    
    print(text)

# 赤色の検出
def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,64,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,64,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）    
    mask = mask1 + mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

def resize():
    #img = cv2.imread('screen_create.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('screen_create.png', cv2.IMREAD_COLOR)
    print(img.shape, img[393][443]) # player2 % color
    print(img.shape, img[209][440]) # player1 % color
    # 0% (255,255,255) 142%(212, 0, 37) 201%(163,0,46)
    print(img.shape, img[290][196]) # player1 % color
    print(img.shape, img[190][433]) # player2 % color

    # player1 damage (174, 414) to (217,455)
    #img = cv2.resize(img, dsize=(512, 512))
    # get damage
    #img = img[418:440, 178:191] # player damage 1  1個目がy2個目がx
    img = img[418:440, 358:373] # player damage 2  1個目がy2個目がx
    cv2.imwrite("player1_damage.png", img)
    # for obs
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize=(128, 128))
    #simg, _ = detect_red_color(img)
    cv2.imshow("test", img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

def mode():
    img = cv2.imread('74.png', cv2.IMREAD_COLOR)
    # almost black to black (0,0,0)
    lower = np.array([0,0,0]) 
    upper = np.array([100,100,100])
    img_mask = cv2.inRange(img, lower, upper)
    img_mask = cv2.bitwise_not(img_mask,img_mask)
    img = cv2.bitwise_and(img, img, mask=img_mask)
    u, counts = np.unique(img[:, :, 0], return_counts=True)
    b = u[np.argmax(counts[1:])]
    u, counts = np.unique(img[:, :, 1], return_counts=True)
    g = u[np.argmax(counts[1:])]
    u, counts = np.unique(img[:, :, 2], return_counts=True)
    r = u[np.argmax(counts[1:])]
    damage = _rgb_to_damage((r,g,b))
    print(damage)
    # 504, 485, 456, 421, 390, 362, 328, 308, 295, 277, , 255, , 237,  ,, 173 (200%)
def _rgb_to_damage(rgb):
    print(rgb)
    (r, g, b) = rgb
    # damage color list 0~150%   // color=R+G
    damage_color_list = [510, 500, 480, 455, 420, 390, 360, 330, 300, 290, 275, 265, 255, 248, 237, 227]
    color = int(r) + int(g)
    damage = 0
    if color >= damage_color_list[0]: return 0
    if color <= damage_color_list[len(damage_color_list)-1]: return 150
    idx = np.abs(np.asarray(damage_color_list) - color).argmin()
    if color >= damage_color_list[idx]:
        rate = (color - damage_color_list[idx]) / (damage_color_list[idx-1] - damage_color_list[idx])
        damage = (idx- 1*rate) * 10 
    else:
        rate = (color - damage_color_list[idx+1]) / (damage_color_list[idx] - damage_color_list[idx+1])
        damage = (idx + 1*rate) * 10 
    return damage

def damage_buffer():
    from collections import deque
    d = deque([], 3)
    d.append(0)
    d.append(1)
    d.append(1)
    d.append(3)
    print(d.count(1)+d.count(0) >= int(len(d)/2)+1)

#resize()
#mode()
damage_buffer()