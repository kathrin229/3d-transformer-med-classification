import random

# random DOWN sampling:
def random_down_sampling(list_dir):
    indices = list(range(0, len(list_dir)))
    good_indices = random.sample(indices, 32)
    good_indices.sort()
    list_dir_32 = [list_dir[i] for i in good_indices]
    return list_dir_32

# symmetrical DOWN sampling:
def symmetrical_down_sampling(list_dir):
    m = len(list_dir)
    k = int(m/32)
    if m % 2 == 1:
        idx = int(m / 2) #CHANGE: not start at index 1 but index in the middle
    else:
        idx = int(m / 2)# - 1 #CHANGE: not start at index -1 but at indes in the middle -1
    list_indices = []
    for i in range(32):
        idx = idx + pow((-1), i) * i * k
        list_indices.append(idx)
    list_indices.sort()
    list_dir_32 = [list_dir[i] for i in list_indices]
    return list_dir_32
