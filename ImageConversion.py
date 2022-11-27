from this import d
import cv2
import numpy as np
from PIL import Image
import math
import datetime

class ImageConversion:
    def __init__(self, imagePath, destanceToCeiling):
        self.imagePath = imagePath
        self.img = Image.open(self.imagePath)

        # 360IMG
        self.rgbImg = self.img.convert('RGB')

        # 画像サイズ(360度画像)
        self.size = self.rgbImg.size

        # 天井までの距離(d)
        self.destanceToCeiling = self.getDestanceToCeiling(destanceToCeiling)

        # 中心の座標(画像視点)
        self.centerCoordinatesX = self.size[0] / 2
        self.centerCoordinatesY = self.size[1] / 2

        self.displayImgInfo()

    def displayImgInfo(self):
        print('-------------------------------------------------')
        print('|| Img info ||')
        print('destance to ceiling: ' + str(self.destanceToCeiling))
        print('size: ' + str(self.size))
        print('CenterCoordinatesX: ' + str(self.centerCoordinatesX) + ' | ' + 'CenterCoordinatesY: ' + str(self.centerCoordinatesY))
        print('-------------------------------------------------')

    def outPutImg(self, img):
        img.save('./output2dImg/2DImg.png')

    # ========================================================================================= #
    # mの天井までの距離をpxに直す
    # ========================================================================================= #
    def getDestanceToCeiling(self, d):
        actualImgWidth = (d*2) * math.pi
        oneMetersPx = self.size[0] / actualImgWidth
        return round(oneMetersPx * d)

    # ========================================================================================= #
    # 2D画像のX,Y座標から極座標のφを求める
    # ========================================================================================= #
    def getPhi(self, x, y):
        inTheSquareRoot =  (x**(2) + y**(2))
        phi = np.arctan(math.sqrt(inTheSquareRoot) / self.destanceToCeiling)
        return phi

    # ========================================================================================= #
    # 2D画像のX,Y座標から極座標のΘを求める
    # ========================================================================================= #
    def getTheta(self, x, y):
        # x座標が0の時、Θは0である
        if x == 0 : return 0

        theta = np.arctan(y / x)
        return theta

    # ========================================================================================= #
    # 取得したい2D画像のx, y座標を引数に指定することで、対応する360度画像のx座標を返す
    # ========================================================================================= #
    def getReferenceCoordinateX(self, x, y):

        CenterPerspectiveCoordinates = self.getCenterPerspectiveCoordinate(x, y)

        theta = self.getTheta(CenterPerspectiveCoordinates[0], CenterPerspectiveCoordinates[1])

        w = self.size[0]

        # 本来のΘの値域とarctanの値域が異なるため値域の調整
        if x <= self.centerCoordinatesX : return w * ((theta+(math.pi/2)) / (2*math.pi))

        if x > self.centerCoordinatesX : return w * ((theta+(3/2*math.pi)) / (2*math.pi))

        return False

    # ========================================================================================= #
    # 取得したい2D画像のx, y座標を引数に指定することで、対応する360度画像のY座標を返す
    # ========================================================================================= #
    def getReferenceCoordinateY(self, x, y):

        CenterPerspectiveCoordinates = self.getCenterPerspectiveCoordinate(x, y)

        phi = self.getPhi(CenterPerspectiveCoordinates[0], CenterPerspectiveCoordinates[1])
        h = self.size[1]

        return (h * (phi / math.pi))

    # ========================================================================================= #
    # 画像視点の座標から中心視点の座標を取得する
    # ========================================================================================= #
    def getCenterPerspectiveCoordinate(self, x, y):
        return [x-self.centerCoordinatesX, -y+self.centerCoordinatesY]

    # ========================================================================================= #
    # 360度画像から2D画像を生成
    # ========================================================================================= #
    def create2DImg(self):
        img2D = Image.new('RGBA', self.size)

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                img2DCoordinateX = round(self.getReferenceCoordinateX(x, y))
                img2DCoordinateY = round(self.getReferenceCoordinateY(x, y))

                r, g, b = self.rgbImg.getpixel((img2DCoordinateX, img2DCoordinateY))

                img2D.putpixel((x, y),(r, g, b))

        # 画像を保存(2D)
        self.outPutImg(img2D)

        # 画像を表示(2D)
        img2D.show()

if __name__ == '__main__':
    imageConversion = ImageConversion('360DegreeImg/test5_d_250cm.jpeg', 2.5)
    imageConversion.create2DImg()