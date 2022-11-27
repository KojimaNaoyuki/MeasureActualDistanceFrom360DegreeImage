import subprocess
import sys

def displayLog(result):
    if result.returncode == 0:
        print('process success')
    else:
        print('process failure')
        print('result: ' + str(result.returncode))
    print('# ------------------------------------------- #')

def main(originalImgPath, d):
    imageConversionFast = subprocess.run('python ImageConversionFast.py ' + originalImgPath + ' ' + d)

    displayLog(imageConversionFast)

    distanceMeasuement = subprocess.run('python DistanceMeasurement.py ' + d)

    displayLog(distanceMeasuement)

if __name__ == '__main__':
    args = sys.argv

    # 360度画像パス
    originalImgPath = args[1]

    # 天井までの距離(d[m])
    d = args[2]

    main(originalImgPath, d)