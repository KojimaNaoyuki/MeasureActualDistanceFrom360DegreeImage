import subprocess
import sys
from os import X_OK
from re import A
from this import d
import cv2
import math
import time

class DistanceMeasurement:
    def __init__(self, imgPath, resizeNum, d):
        self.colorimg = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        if self.colorimg is None : return -1

        # 画像読み込み
        self.colorimg = self.scale_to_width(self.colorimg, resizeNum)

        # クリック座標
        self.selectCoordinate = []

        # 1pxの実際の距離[mm]
        self.onePxActualDistance = (1.168 * d)

        self.displayImg(self.colorimg)

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN: 
            self.selectCoordinate.append([x, y])

            cv2.circle(self.colorimg, (x, y), 5, color=(255, 0, 0), thickness=-1)
            
            print(x, y)

            if len(self.selectCoordinate) == 4:
                actualLength = self.getActualLength(self.selectCoordinate)
                self.selectCoordinate.clear()
                print('------------------------------------')
                print(str(actualLength[0]) + "cm")
                print(str(actualLength[1]) + "cm")
                print(str(actualLength[2]) + "cm")
                print(str(actualLength[3]) + "cm")
                print("斜辺: " + str(actualLength[4]) + "cm")
                print('------------ 面積 ------------')
                print(str(self.getArea(actualLength)) + "cm^2")
                print(str(self.getArea(actualLength) / 10000) + "m^2")
                print('------------------------------------')

                # 面積(m^2をファイルに書き込み)
                self.inputPlaceAreaValueToFile(str(self.getArea(actualLength) / 10000))

                time.sleep(1)

                sys.exit()
    
    # ========================================================================================= #
    # 画像のリサイズ
    # ========================================================================================= #
    def scale_to_width(self, img, width):
        h, w = img.shape[:2]
        height = round(h * (width / w))
        dst = cv2.resize(img, dsize=(width, height))

        return dst

    # ========================================================================================= #
    # クリックした長さを取得(実距離)
    # ========================================================================================= #
    def getActualLength(self, selectCoordinate):
        # 対角線の座標
        diagonalCoordinate = self.getDiagonalCoordinate(selectCoordinate)

        # 対角線の座標の格納index検索
        index1 = selectCoordinate.index(diagonalCoordinate[0])
        index2 = selectCoordinate.index(diagonalCoordinate[1])
        print("diagonalCoordinate index: " + str(index1) + " | " + str(index2))

        pxLength = []

        # 一辺の長さを取得[px](4辺 対角線にはならないように)
        for index, currentSelectCoordinate in enumerate(selectCoordinate):
            if index1 == index or index2 == index: continue

            pxLength.append(self.getPxLength(selectCoordinate[index1], currentSelectCoordinate))
            pxLength.append(self.getPxLength(selectCoordinate[index2], currentSelectCoordinate))

        # 斜辺の長さを取得[px]
        pxLengthHypotenuse = self.getPxLength(diagonalCoordinate[0], diagonalCoordinate[1])

        return [
            pxLength[0] * self.onePxActualDistance, 
            pxLength[1] * self.onePxActualDistance, 
            pxLength[2] * self.onePxActualDistance, 
            pxLength[3] * self.onePxActualDistance, 
            pxLengthHypotenuse * self.onePxActualDistance
            ]

    # ========================================================================================= #
    # 対角線の座標を取得する
    # ========================================================================================= #
    def getDiagonalCoordinate(self, selectCoordinate):

        if( self.judgDiagonal(selectCoordinate[0], selectCoordinate[1], selectCoordinate[2], selectCoordinate[3]) ): 
            return [selectCoordinate[0], selectCoordinate[1]]

        if( self.judgDiagonal(selectCoordinate[1], selectCoordinate[2], selectCoordinate[0], selectCoordinate[3]) ):
            return [selectCoordinate[1], selectCoordinate[2]]

        if( self.judgDiagonal(selectCoordinate[2], selectCoordinate[3], selectCoordinate[0], selectCoordinate[1]) ):
            return [selectCoordinate[2], selectCoordinate[3]]

        if( self.judgDiagonal(selectCoordinate[0], selectCoordinate[2], selectCoordinate[1], selectCoordinate[3]) ):
            return [selectCoordinate[0], selectCoordinate[2]]

        if( self.judgDiagonal(selectCoordinate[1], selectCoordinate[3], selectCoordinate[0], selectCoordinate[2]) ):
            return [selectCoordinate[1], selectCoordinate[3]]

        return False

    def judgDiagonal(self, diagonalCandidate1, diagonalCandidate2, verificationCoordinates1, verificationCoordinates2):
        a = abs(diagonalCandidate1[1] - diagonalCandidate2[1]) / abs(diagonalCandidate1[0] - diagonalCandidate2[0])
        b = diagonalCandidate1[1] - (a * diagonalCandidate1[0])

        judgNumber1 = verificationCoordinates1[1] - (a * verificationCoordinates1[0]) - b
        judgNumber2 = verificationCoordinates2[1] - (a * verificationCoordinates2[0]) - b

        if judgNumber1*judgNumber2 < 0:
            return True

        return False

    # ========================================================================================= #
    # クリックした長さを取得(ピクセル)
    # ========================================================================================= #
    def getPxLength(self, point1, point2):
        X = 0
        Y = 1

        # リサイズを戻すための倍率
        resizeMagnification = 4

        side1 = abs(point1[X] - point2[X])
        side2 = abs(point1[Y] - point2[Y])

        return math.sqrt( (side1**2) + (side2**2) ) * resizeMagnification

    # ========================================================================================= #
    # 面積を取得する
    # ========================================================================================= #
    def getArea(self, actualLength):
        diagonalCoordinate = actualLength[4]
        
        s = (actualLength[0] + actualLength[1] + diagonalCoordinate) / 2
        area1 = math.sqrt(s * ( (s-actualLength[0])*(s-actualLength[1])*(s-diagonalCoordinate) ))

        s = (actualLength[2] + actualLength[3] + diagonalCoordinate) / 2
        area2 = math.sqrt(s * ( (s-actualLength[2])*(s-actualLength[3])*(s-diagonalCoordinate) ))

        return area1 + area2

    # ========================================================================================= #
    # 画像の表示
    # ========================================================================================= #
    def displayImg(self, img):
        firstFlag = True

        while True:
            cv2.imshow('IMG', img)

            if firstFlag : cv2.setMouseCallback('IMG', self.onMouse)
            firstFlag = False

            cv2.waitKey(1)

    # ========================================================================================= #
    # 面積をファイルに書き込み
    # ========================================================================================= #
    def inputPlaceAreaValueToFile(self, area):
        f = open('../CongestionStatusGraspScript/data/placeAreaValue.txt', 'w')
        f.write(area)
        f.close()
    
# ========================================================================================= #
# 設定ファイル読み込み
# ========================================================================================= #
def readConfig():
    f = open('./config/resize.txt', 'r')
    resizeNum = f.readlines()
    f.close()
    return resizeNum

if __name__ == '__main__':
    args = sys.argv

    resizeNum = readConfig()

    # d = ?[m]
    cornerCandidateSearch = DistanceMeasurement('./output2dImg/2DImg.png', int(resizeNum[0]), float(args[1]))