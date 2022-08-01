
def delete_empty_image(path):
    im = cv.imread(path)
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        os.remove(path)

def resize_all_images_512():
    rootpath = './data/dataset_seg'
    labels = ['CP', 'NCP', 'Normal']

    img_size = 512

    for label in labels:
        rootpath_label = rootpath + '/' + label

        for directory in sorted(os.listdir(rootpath_label)):
            subpath = os.path.join(rootpath_label, directory)

            for subdirectory in sorted(os.listdir(subpath)):
                subsubpath = os.path.join(subpath, subdirectory)

                for image_file in sorted(os.listdir(subsubpath)):
                    if image_file == '.ipynb_checkpoints':
                        break
                    img_path = os.path.join(subsubpath, image_file)

                    im = cv.imread(img_path)
                    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

                    if img.shape[0] != img_size:
                        down_width = img_size
                        img_resize = cv.resize(img, (down_width, down_width), interpolation= cv.INTER_LINEAR)
                        cv.imwrite(img_path, img_resize) 

 