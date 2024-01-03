'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，可以参考一下txt_annotation.py文件。
'''
from PIL import Image
from classification import Classification
import time

classfication = Classification()

# while True:
img = '/root/autodl-tmp/classification-pytorch/datasets/test/YES/YES_1469.png'

try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
else:
    # start = time.time()
    class_name = classfication.detect_image(image)
    # end = time.time()
    print(class_name)
    # print('程序运行时间:%s毫秒' % ((end - start)*1000))