import numpy as np
import matplotlib.pyplot as plt


# mean_std = ([0.449], [1.0])
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class_color = np.array([[255, 255, 255], [0, 0, 0]])


def view_image(img, idx, tensor, save=False):
    if not tensor:
        print(img.mode)
        # pil2numpy
        img = np.asarray(img)
        print(img.shape)
    else:
        # input shape = (RGB, H, W)
        # tensor2numpy
        img = img.numpy().transpose((1, 2, 0))
        for i in range(3):
            img[:, :, i] = (mean_std[1][i] * img[:, :, i] + mean_std[0][i])
        img = img*255.0
    # print(img[0, :])
    h, w, c = img.shape
    f1 = plt.figure(1, figsize=(w, h), dpi=1)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.imshow(img)
    if save:
        # it is very important to save before show for grey-scale image
        if not tensor:
            plt.savefig("./%04d_raw_image.jpg"%(idx))
        else:
            plt.savefig("./%04d_trans_image.jpg"%(idx))
    else:
        plt.show()
    plt.close()


def view_grey_image(img, idx, tensor, save=False):
    if not tensor:
        print(img.mode)
        # pil2numpy
        img = np.asarray(img)
        print(img.shape)
    else:
        # input shape = (1, H, W)
        # tensor2numpy
        img = img.numpy().transpose((1, 2, 0))
        img = (mean_std[1][0] * img + mean_std[0][0])*255.0
        img = img.squeeze()
    # print(img[0, :])
    h, w = img.shape
    f1 = plt.figure(1, figsize=(w, h), dpi=1)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    if save:
        # it is very important to save before show for grey-scale image
        if not tensor:
            plt.savefig("./%04d_raw_image.jpg"%(idx))
        else:
            plt.savefig("./%04d_trans_image.jpg"%(idx))
    else:
        plt.show()
    plt.close()


def view_gt(img, idx, tensor, save=False):
    if not tensor:
        print(img.mode)
        # pil2numpy
        img = np.asarray(img)
        # print("not tensor", img[0, :])
    else:
        # input size = (H, W)
        # tensor2numpy
        img = img.numpy().astype(np.uint8)
        # print("tensor", img[0, :])
    # print(img.shape)
    # convert binary map with single channel to color map
    r = img.copy()
    g = img.copy()
    b = img.copy()
    for l in range(0,2):
        r[img==l] = class_color[l,0]
        g[img==l] = class_color[l,1]
        b[img==l] = class_color[l,2]

    img = np.dstack([r, g, b])
    h, w, c = img.shape
    f1 = plt.figure(1, figsize=(w, h), dpi=1)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.imshow(img)
    if save:
        # it is very important to save before show for grey-scale image
        if not tensor:
            plt.savefig("./%04d_raw_gt.jpg"%(idx))
        else:
            plt.savefig("./%04d_trans_gt.jpg"%(idx))
    else:
        plt.show()
    plt.close()
