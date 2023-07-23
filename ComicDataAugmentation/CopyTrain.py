import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lq_folder_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_lq_validation/'
    hr_folder_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_train/'
    dest_folder_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_validation/'

    lq_folder_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_lq_test/'
    hr_folder_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_train/'
    dest_folder_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_test/'

    for filename in os.listdir(lq_folder_validation):
        current = os.path.join(hr_folder_validation, filename)
        if not os.path.isfile(current):
            current = current.replace('.jpg', '.png')
        new = os.path.join(dest_folder_validation, filename)
        os.rename(current, new)
        print(new)

    for filename in os.listdir(lq_folder_test):
        current = os.path.join(hr_folder_test, filename)
        if not os.path.isfile(current):
            current = current.replace('.jpg', '.png')
        new = os.path.join(dest_folder_test, filename)
        os.rename(current, new)
        print(new)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
