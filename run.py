import subprocess
import sys

def displayLog(result):
    if result.returncode == 0:
        print('process success')
    else:
        print('process failure')
        print('result: ' + str(result.returncode))
    print('# ------------------------------------------- #')

def main(originalImgPath, d, reductionRatio):
    imageConversionFast = subprocess.run(['python3 /opt/MeasureActualDistanceFrom360DegreeImage/ImageConversionFast.py ' + originalImgPath + ' ' + d + ' ' + reductionRatio], shell=True)

    displayLog(imageConversionFast)

    distanceMeasuement = subprocess.run(['python3 /opt/MeasureActualDistanceFrom360DegreeImage/DistanceMeasurement.py ' + d + " " + reductionRatio], shell=True)

    displayLog(distanceMeasuement)

if __name__ == '__main__':
    args = sys.argv

    # 360度画像パス
    originalImgPath = args[1]

    # 天井までの距離(d[m])
    d = args[2]
    
    # 縮小率
    reductionRatio = args[3]

    main(originalImgPath, d, reductionRatio)
