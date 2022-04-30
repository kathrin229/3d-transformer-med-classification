import os

rootpath_CP = 'data/dataset_seg/CP'
rootpath_NCP = 'data/dataset_seg/NCP'
rootpath_Normal = 'data/dataset_seg/Normal'

list_CP = []
list_NCP = []
list_Normal = []

list_CP_patients = []
list_NCP_patients = []
list_Normal_patients = []

for directory in sorted(os.listdir(rootpath_CP)):
    subpath = os.path.join(rootpath_CP, directory)
    list_CP_patients.append(len(os.listdir(subpath)))
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_CP.append(len(os.listdir(subsubpath))) 
        
for directory in sorted(os.listdir(rootpath_NCP)):
    subpath = os.path.join(rootpath_NCP, directory)
    list_NCP_patients.append(len(os.listdir(subpath)))
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_NCP.append(len(os.listdir(subsubpath)))

for directory in sorted(os.listdir(rootpath_Normal)):
    subpath = os.path.join(rootpath_Normal, directory)
    list_Normal_patients.append(len(os.listdir(subpath)))
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_Normal.append(len(os.listdir(subsubpath)))        

print("Common Pneumonia scans: " + str(len(list_CP)))
print("Novel Coronavirus Pneumonia scans: " + str(len(list_NCP)))
print("Normal scans: " + str(len(list_Normal)))

print(sum(list_CP))
print(sum(list_NCP))
print(sum(list_Normal))

print("Patients CP: " + str(len(list_CP_patients)))
print("Patients NCP: " + str(len(list_NCP_patients)))
print("Patients Normal: " + str(len(list_Normal_patients)))

print(sum(i > 50 for i in list_CP))
print(sum(i > 50 for i in list_NCP))
print(sum(i > 50 for i in list_Normal))

print(sum(i > 50 for i in list_CP)/len(list_CP))
print(sum(i > 50 for i in list_NCP)/len(list_NCP))
print(sum(i > 50 for i in list_Normal)/len(list_Normal))

print(sum(i > 32 for i in list_CP))
print(sum(i > 32 for i in list_NCP))
print(sum(i > 32 for i in list_Normal))

print(1-sum(i > 32 for i in list_CP)/len(list_CP))
print(1-sum(i > 32 for i in list_NCP)/len(list_NCP))
print(1-sum(i > 32 for i in list_Normal)/len(list_Normal))

print('done')