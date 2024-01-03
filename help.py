import os
import cv2
import torchvision.models as models

##按标签名重命名图片，  新名字：YES/NO_序号.png
# base_paths = os.listdir("datasets")
# for base_path in base_paths:
#     base_path = os.path.join("datasets",base_path)
#     sets = os.listdir(base_path)
#     for set in sets:
#         path = os.path.join(base_path,set)
#         print(path)
#         names = os.listdir(path)
#         for name in names:
#             src = os.path.join(path,name)
#             name = set + "_" + name.split("_")[1]
#             dst = os.path.join(path,name)
#             os.rename(src,dst)


# ##resize图片
# base_paths = os.listdir("datasets")
# for base_path in base_paths:
#     base_path = os.path.join("datasets",base_path)
#     sets = os.listdir(base_path)
#     for set in sets:
#         path = os.path.join(base_path,set)
#         names = os.listdir(path)
#         for name in names:
#             img = cv2.imread(os.path.join(path,name),0)
#             img = cv2.resize(img,(512,512))
#             cv2.imwrite(os.path.join(path,name),img)


##查看官方网络结构及名称
# net = models.mobilenet_v3_small(pretrained=True)
# print(net)
# net = models.mobilenet_v2(pretrained=True)
# print(net)
net = models.shufflenet_v2_x1_0(pretrained=True)
print(net)

