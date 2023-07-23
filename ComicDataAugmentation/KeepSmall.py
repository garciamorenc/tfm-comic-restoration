import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lq_folder_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_lq_validation/'
    hr_folder_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_validation/'

    for filename in os.listdir(hr_folder_validation):
        current = os.path.join(hr_folder_validation, filename)
        if not filename.__contains__('T3'):
            os.remove(current)
            print(current)

    for filename in os.listdir(lq_folder_validation):
        current = os.path.join(lq_folder_validation, filename)
        if not filename.__contains__('T3'):
            os.remove(current)
            print(current)

        # See PyCharm help at https://www.jetbrains.com/help/pycharm/
