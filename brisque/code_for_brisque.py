import torch
import piq
from skimage.io import imread


@torch.no_grad()
def main():
    # Read RGB image and it's noisy version
    directory = 'test'
    im = torch.tensor(imread(directory + '/image.bmp')).permute(2, 0, 1)[None, ...] / 255.
    im1 = torch.tensor(imread(directory + '/image.jpg')).permute(2, 0, 1)[None, ...] / 255.
    #im2 = torch.tensor(imread(directory + '/image2.bmp')).permute(2, 0, 1)[None, ...] / 255.
    #im3 = torch.tensor(imread(directory + '/image3.bmp')).permute(2, 0, 1)[None, ...] / 255.
    #im4 = torch.tensor(imread(directory + '/image4.bmp')).permute(2, 0, 1)[None, ...] / 255.
    #im5 = torch.tensor(imread(directory + '/image5.bmp')).permute(2, 0, 1)[None, ...] / 255.

    im_number = 0
    for image in [im, im1]:   #, im2, im3, im4, im5]:
        print('im' + str(im_number))
        im_number += 1

        # To compute BRISQUE score as a measure, use lower case function from the library
        brisque_index: torch.Tensor = piq.brisque(image, data_range=1., reduction='sum')
        # In order to use BRISQUE as a loss function, use corresponding PyTorch module.
        # Note: the back propagation is not available using torch==1.5.0.
        # Update the environment with latest torch and torchvision.
        brisque_loss: torch.Tensor = piq.BRISQUELoss(data_range=1., reduction='none')(image)
        print(f"BRISQUE index: {brisque_index.item():0.4f}, loss: {brisque_loss.item():0.4f}")
        print('-------------------------------------------------------------------------------------')

    # To compute TV as a measure, use lower case function from the library:
    ##tv_index: torch.Tensor = piq.total_variation(x)
    # In order to use TV as a loss function, use corresponding PyTorch module:
    ##tv_loss: torch.Tensor = piq.TVLoss(reduction='none')(x)
    ##print(f"TV index: {tv_index.item():0.4f}, loss: {tv_loss.item():0.4f}")

if __name__ == '__main__':
    main()

