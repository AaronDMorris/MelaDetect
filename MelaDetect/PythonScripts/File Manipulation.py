import shutil
import os, random

path = "E:\\MelaDetect\\alldata_resized_benign"

for f in range(0, 1899):
    file = random.choice(os.listdir(path))
    shutil.move(path + "\\" + file, "E:\\MelaDetect\\alldata_testdata_benign\\" + file)
    
    
    
import fnmatch    
    
iterator = 1    
'''
for file in os.listdir(path):
        os.rename(path + file, str(iterator) + ".jpg")
        iterator += 1
'''    