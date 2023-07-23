
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    crop_pad_size = 600
    h = 540
    w = 648
    pad_h = max(0, crop_pad_size - h)
    pad_w = max(0, crop_pad_size - w)
    if pad_w == 0:
        pad_w = crop_pad_size
    print(pad_h, pad_w)




