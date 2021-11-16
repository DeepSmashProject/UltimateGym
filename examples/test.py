import os
from PIL import Image
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
    img = cv2.imread('screenshot3.png', cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread('screenshot3.png', cv2.IMREAD_COLOR)
    #print(img)
    img = cv2.resize(img, dsize=(84, 84))
    #simg, _ = detect_red_color(img)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


resize()