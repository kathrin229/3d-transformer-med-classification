import os

rootpath_CP = '../../data/dataset_seg/CP'
rootpath_NCP = '../../data/dataset_seg/NCP'
rootpath_Normal = '../../data/dataset_seg/Normal'

list_CP = []
list_NCP = []
list_Normal = []

for directory in sorted(os.listdir(rootpath_CP)):
    subpath = os.path.join(rootpath_CP, directory)
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_CP.append(len(os.listdir(subsubpath)))
        
for directory in sorted(os.listdir(rootpath_NCP)):
    subpath = os.path.join(rootpath_NCP, directory)
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_NCP.append(len(os.listdir(subsubpath)))
        
for directory in sorted(os.listdir(rootpath_Normal)):
    subpath = os.path.join(rootpath_Normal, directory)
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_Normal.append(len(os.listdir(subsubpath)))

print("Common Pneumonia samples: " + str(len(list_CP)))
print("Novel Coronavirus Pneumonia samples: " + str(len(list_NCP)))
print("Normal samples: " + str(len(list_Normal)))

print(sum(i > 50 for i in list_CP))
print(sum(i > 50 for i in list_NCP))
print(sum(i > 50 for i in list_Normal))

print(sum(i > 50 for i in list_CP)/len(list_CP))
print(sum(i > 50 for i in list_NCP)/len(list_NCP))
print(sum(i > 50 for i in list_Normal)/len(list_Normal))