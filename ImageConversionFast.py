# ================================================= #
# ImageConversion.pyを高速化することを目指したプログラム
# 7s程度早くなる(自宅環境)
# ================================================= #

import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
import math

args = sys.argv

img = Image.open(args[1])

# 360IMG
rgbImg = img.convert('RGB')

# 360IMG 配列 (行列のためx, yが反転)
rgbImgArray = np.asarray(rgbImg)

# 画像サイズ(360度画像)
size = rgbImg.size

# d(m)
d = float(args[2])

# 天井までの距離(d)
actualImgWidth = (d*2) * math.pi
oneMetersPx = size[0] / actualImgWidth
destanceToCeiling = round(oneMetersPx * d)

# 中心の座標(画像視点)
centerCoordinatesX = size[0] / 2
centerCoordinatesY = size[1] / 2

img2D = Image.new('RGB', size)

img2DArray = np.array(img2D)

for x in range(size[0]):
    for y in range(size[1]):
        # 画像視点の座標から中心視点の座標を取得する
        CenterPerspectiveCoordinates = [x-centerCoordinatesX, -y+centerCoordinatesY]

        # ----------------------------------------------------------------------------- #
        # 取得したい2D画像のx, y座標を引数に指定することで、対応する360度画像のx座標を返す
        # ----------------------------------------------------------------------------- #
        # 2D画像のX,Y座標から極座標のΘを求める
        if CenterPerspectiveCoordinates[0] != 0:
            theta = np.arctan(CenterPerspectiveCoordinates[1] / CenterPerspectiveCoordinates[0])

            w = size[0]

            # 本来のΘの値域とarctanの値域が異なるため値域の調整
            if x <= centerCoordinatesX : img2DCoordinateX = round(w * ((theta+(math.pi/2)) / (2*math.pi)))

            if x > centerCoordinatesX : img2DCoordinateX = round(w * ((theta+(3/2*math.pi)) / (2*math.pi)))
        else:
            img2DCoordinateX = 0
        # ----------------------------------------------------------------------------- #
        
        # ----------------------------------------------------------------------------- #
        # 取得したい2D画像のx, y座標を引数に指定することで、対応する360度画像のY座標を返す
        # ----------------------------------------------------------------------------- #
        # 2D画像のX,Y座標から極座標のφを求める
        inTheSquareRoot =  (CenterPerspectiveCoordinates[0]**(2) + CenterPerspectiveCoordinates[1]**(2))
        phi = np.arctan(math.sqrt(inTheSquareRoot) / destanceToCeiling)

        h = size[1]

        img2DCoordinateY = round(h * (phi / math.pi))
        # ----------------------------------------------------------------------------- #

        r, g, b = rgbImgArray[img2DCoordinateY, img2DCoordinateX, :]

        img2DArray[y][x] = [r, g, b]


# 画像化
img2D = Image.fromarray(img2DArray)

# 画像を保存(2D)
img2D.save('./output2dImg/2DImg.png')

# 画像を表示(2D)
# img2D.show()
