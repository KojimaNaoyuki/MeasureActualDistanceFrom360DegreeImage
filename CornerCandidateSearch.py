from re import S
import cv2
import numpy as np
from numpy import append, linalg as LA
import sys
import math

class CornerCandidateSearch:
    def __init__(self, colorImagePath, depthImagePath, resizeNum):
        colorimg = cv2.imread(colorImagePath, cv2.IMREAD_COLOR)
        depthImage = cv2.imread(depthImagePath, cv2.IMREAD_COLOR)
        if colorimg is None : return -1

        self.colorimg = self.scale_to_width(colorimg, resizeNum)
        self.depthImage = self.scale_to_width(depthImage, resizeNum)

        self.imgSizeH, self.imgSizeW, channels = self.colorimg.shape[:3]

        self.centerCoordinatesX = self.imgSizeW / 2
        self.centerCoordinatesY = self.imgSizeH / 2

        # 画像のコントラストと明るさを調節
        self.colorimg = self.adjust(self.colorimg, 1.4, 0)

        # グレースケールに変換
        imageGray = cv2.cvtColor(self.colorimg.copy(), cv2.COLOR_BGR2GRAY)

        # 白黒反転(これにより精度上昇)
        self.imageReversed = cv2.bitwise_not(imageGray)

    def run(self, debugFlag, detectionMethod):
        if detectionMethod == 'LSD':
            # LSD
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

            # ライン取得 LSD
            lines, width, prec, nfa = lsd.detect(self.imageReversed)

            # 画像に検出した直線を合成
            self.imgLine = lsd.drawSegments(self.colorimg, lines)

        if detectionMethod == 'Hough':
            # ハフ変換
            lines = cv2.HoughLinesP(self.imageReversed, rho=1, theta=np.pi/360, threshold=100, minLineLength=400, maxLineGap=5)

            # 直線を描画
            for line in lines:
                x1, y1, x2, y2 = line[0]

                self.colorimg = cv2.line(self.colorimg, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if detectionMethod == 'FLD':
            # FLD
            fld = cv2.ximgproc.createFastLineDetector(20, 1.41421356, 50, 50, 3, False)

            # ライン取得 FLD
            lines = fld.detect(self.imageReversed)

            # 画像に検出した直線を合成
            self.imgLine = fld.drawSegments(self.colorimg, lines)

        cv2.imshow('line', self.imgLine)
        cv2.waitKey(0)

        # 角候補の中で最も深い点を取得
        self.serachDeepestCorner(lines)

        # 角候補を取得
        filteredCoordinates = self.serachCorner(self.imgLine, lines)

        # 角候補から角を決定
        decisionCornerCoordinates = self.decisionCorner(filteredCoordinates)

        self.drawCorner(self.imgLine, decisionCornerCoordinates)

        self.writeFile(decisionCornerCoordinates)

        # 画像の保存
        cv2.imwrite("./outputResultImg/FilteringImg.jpg", self.imgLine)
        
        if debugFlag == True:
            # 検出線と処理画像の合成表示
            cv2.imshow("Fast Line Detector", self.imgLine)
            cv2.setMouseCallback('Fast Line Detector', self.onMouse)
            cv2.waitKey(0)

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN : print(x, y)

    def writeFile(self, data):
        f = open('./data/cornerCoordinates.txt', 'w')

        for item in data:
            f.write(item + '\n')

        f.close()
    
    # ========================================================================================= #
    # 画像の彩度の変更
    # ========================================================================================= #
    def chengeSaturation(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_magnification = 3.5  # 彩度(Saturation)の倍率
        v_magnification = 1  # 明度(Value)の倍率

        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*s_magnification  # 彩度の計算
        img_hsv[:,:,(2)] = img_hsv[:,:,(2)]*v_magnification  # 明度の計算
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)  # 色空間をHSVからBGRに変換

        cv2.imshow('origin img', img)
        cv2.imshow('chengeed img', img_bgr)
        cv2.waitKey(0)

        return img_bgr

    # ========================================================================================= #
    # 画像のコントラストと明るさを調節
    # ========================================================================================= #
    def adjust(self, img, alpha, beta):
        # 積和演算を行う。
        dst = alpha * img + beta
        # [0, 255] でクリップし、uint8 型にする。
        return np.clip(dst, 0, 255).astype(np.uint8)

    # ========================================================================================= #
    # 画像のリサイズ
    # ========================================================================================= #
    def scale_to_width(self, img, width):
        h, w = img.shape[:2]
        height = round(h * (width / w))
        dst = cv2.resize(img, dsize=(width, height))

        return dst

    # ========================================================================================= #
    # 角候補の中で最も深い点の座標と深度を取得
    # ========================================================================================= #
    def serachDeepestCorner(self, lines):
        depthData = {}
        leftFlag = False

        for index, coordinate in enumerate(lines):
            # self.depthImage[y, x] 行列だから
            # LSDからの線の端点座標は1~801で、openCVのimg座標は0~799である可能性が高いため-1している
            print(coordinate)
            depthData[index+1] = [
                self.depthImage[math.floor(coordinate[0][1]-1), math.floor(coordinate[0][0]-1)][0]
                ]
            depthData[index+2] = [
                self.depthImage[math.floor(coordinate[0][3]-1), math.floor(coordinate[0][2]-1)][0]
                ]

        depthData = sorted(depthData.items(), key=lambda x:x[1])

        # 角候補の中で最も深い深度の取得
        self.deepestData = depthData[0][1][0]
        print('deepestData: ' + str(self.deepestData))
        
        deepestIndex = (depthData[0][0] / 2) - 1
        if not deepestIndex.is_integer():
            leftFlag = True
            deepestIndex = math.ceil(deepestIndex)

        # 角候補の中で最も深い深度の座標を取得
        if leftFlag : self.deepestCoordinate = [lines[int(deepestIndex)][0][0], lines[int(deepestIndex)][0][1]]
        if not leftFlag : self.deepestCoordinate = [lines[int(deepestIndex)][0][2], lines[int(deepestIndex)][0][3]]

        return True

    # ========================================================================================= #
    # 角候補を検索
    # ========================================================================================= #
    def serachCorner(self, img, lines):
        count = 0
        coordinates = []
        for coordinate in lines:
            if self.filteringCorner(coordinate[0][0], coordinate[0][1], lines, coordinate) == True:
                count += 1
                print("draw: " + str(coordinate[0][0]) + " | " + str(coordinate[0][1]))
                coordinates.append(str(coordinate[0][0]) + " " + str(coordinate[0][1]))
                cv2.circle(img, (int(coordinate[0][0]), int(coordinate[0][1])), 5, color=(255, 0, 0), thickness=-1)

            if self.filteringCorner(coordinate[0][0], coordinate[0][1], lines, coordinate) == True:
                count += 1
                print("draw: " + str(coordinate[0][2]) + " | " + str(coordinate[0][3]))
                coordinates.append(str(coordinate[0][2]) + " " + str(coordinate[0][3]))
                cv2.circle(img, (int(coordinate[0][2]), int(coordinate[0][3])), 5, color=(255, 0, 0), thickness=-1)
        print("corner: " + str(count))

        return coordinates

    # ========================================================================================= #
    # 最終決定した角を描画
    # ========================================================================================= #
    def drawCorner(self, img, coordinates):
        for coordinate in coordinates:
            tmpCoordinate = coordinate.split()
            cv2.circle(img, (int(float(tmpCoordinate[0])), int(float(tmpCoordinate[1]))), 5, color=(0, 255, 0), thickness=-1)
            print('decision corner: ' + str(tmpCoordinate[0]) + ' | ' + str(tmpCoordinate[1]))

    # ========================================================================================= #
    # 角候補のフィルタリングのラップメソッド
    # ========================================================================================= #
    def filteringCorner(self, tgX, tgY, lines, coordinate):
        if self.filteringLength(coordinate) and self.filteringParallelLine(lines, coordinate) and self.filteringDeepestPlace(lines, coordinate):
            return True
        else:
            return False

    # ========================================================================================= #
    # 角候補のフィルタリング 一定の長さ以下の直線の終端を候補から除外
    # ========================================================================================= #
    def filteringLength(self, coordinate):
        # 許容範囲
        minLength = 20

        lengthX = abs(coordinate[0][0] - coordinate[0][2])
        lengthY = abs(coordinate[0][1] - coordinate[0][3])

        if lengthX <= minLength : return False
        if lengthY <= minLength : return False

        return True

    # ========================================================================================= #
    # 角候補のフィルタリング 一定の近さに他の角候補が存在しない場合に候補から除外
    # ========================================================================================= #
    def filteringDistance(self, tgX, tgY, lines):
        # 許容範囲(px)
        tolerance = 10

        cornerCount = 0

        for coordinate in lines:
            difference1X = abs(tgX - coordinate[0][0])
            difference1Y = abs(tgY - coordinate[0][1])
            difference2X = abs(tgX - coordinate[0][2])
            difference2Y = abs(tgY - coordinate[0][3])

            if difference1X == 0 or difference1Y == 0 or difference2X == 0 or difference2Y == 0 : continue
            
            flag = True
            if not difference1X <= tolerance : flag = False
            if not difference1Y <= tolerance : flag = False
            
            if flag == True : cornerCount += 1
            
            flag = True
            if not difference2X <= tolerance : flag = False
            if not difference2Y <= tolerance : flag = False
            
            if flag == True : cornerCount += 1

            if cornerCount == 2 : return True

        print("filtering: " + str(tgX) + " | " + str(tgY))

        return False

    # ========================================================================================= #
    # 角候補のフィルタリング 許容範囲内の誤差を許し、二つ以上の平行の直線がある線の端点のみ残す
    # ========================================================================================= #
    def filteringParallelLine(self, lines, coordinate):
        # 許容範囲(ラジアン)
        toleranceAngle = math.radians(2)

        baseLine = coordinate

        for fluctuationLine in lines:
            if np.allclose(baseLine, fluctuationLine) : continue
            if self.calculationLineAngle(baseLine[0], fluctuationLine[0]) < toleranceAngle : return True
        
        return False

    # ========================================================================================= #
    # 画像視点の座標Yを数学的視点の座標に変換した値(y)
    # ========================================================================================= #
    def getMathPerspectiveCoordinateY(self, y):
        return -(y-self.imgSizeH)

    # ========================================================================================= #
    # 二直線のなす角を求める
    # ========================================================================================= #
    def calculationLineAngle(self, lineFirst, lineSecond):
        lineFirstVector = [ abs(lineFirst[2] - lineFirst[0]), abs(self.getMathPerspectiveCoordinateY(lineFirst[3]) - self.getMathPerspectiveCoordinateY(lineFirst[1])) ]

        lineSecondVector = [ abs(lineSecond[2] - lineSecond[0]), abs(self.getMathPerspectiveCoordinateY(lineSecond[3]) - self.getMathPerspectiveCoordinateY(lineSecond[1])) ]

        return self.tangent_angle(lineFirstVector, lineSecondVector)

    def tangent_angle(self, u: np.ndarray, v: np.ndarray):
        i = np.inner(u, v)
        n = LA.norm(u) * LA.norm(v)
        c = i / n
        return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

    # ========================================================================================= #
    # 角候補のフィルタリング 許容範囲内の誤差を許し、画像の最も深い場所以外の角候補を除外する
    # ========================================================================================= #
    def filteringDeepestPlace(self, lines, coordinate):
        # 許容範囲(深度)
        tolerance = 20
        
        depthValue = self.depthImage[int(coordinate[0][1]), int(coordinate[0][0])][0]

        if tolerance < (depthValue - self.deepestData):
            return False
        else:
            return True

    # ========================================================================================= #
    # 角決定 中心点から最も遠い4点
    # ========================================================================================= #
    def decisionCorner(self, coordinate):
        # 絞り込む角候補の数
        decisionCornerNum = 4
        distanceFromCenterArr = {}

        for index, currentCoordinate in enumerate(coordinate):
            tmpCoordinate = currentCoordinate.split()
            distanceFromCenter = math.sqrt( ( float(tmpCoordinate[0])-self.centerCoordinatesX )**2 + ( float(tmpCoordinate[1])-self.centerCoordinatesY )**2 )
            distanceFromCenterArr[index] = distanceFromCenter

        distanceFromCenterArr = sorted(distanceFromCenterArr.items(), key=lambda x:x[1], reverse=True)

        distanceFromCenterResultCoordinate = []
        for num in range(decisionCornerNum):
            distanceFromCenterResultCoordinate.append(coordinate[distanceFromCenterArr[num][0]])

        return distanceFromCenterResultCoordinate

# ========================================================================================= #
# 設定ファイル読み込み
# ========================================================================================= #
def readConfig():
    f = open('./config/resize.txt', 'r')
    resizeNum = f.readlines()
    f.close()
    return resizeNum

if __name__ == '__main__':
    resizeNum = readConfig()

    cornerCandidateSearch = CornerCandidateSearch('./output2dImg/2DImg.png', './depthImg/2DImg.png', int(resizeNum[0]))
    cornerCandidateSearch.run(False, 'LSD')