import os

path = r'E:\Project_DL\crawl-img-tool\data\balls'

names = os.listdir(path)

for name in names:
    os.rename(os.path.join(path,name), os.path.join(path,'ball'+name))