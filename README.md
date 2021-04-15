# Contour_aware_unet
A simple reproduced model of thesis "Contour-aware multi-label chest X-ray organ segmentation"

Thesis source [Contour-aware multi-label chest X-ray organ segmentation](https://www.researchgate.net/publication/339122497_Contour-aware_multi-label_chest_X-ray_organ_segmentation)

## Dataset
This model has been tested on [Oxford-pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [ACDC Cardiac 2D](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html), and [CVC-Colon Polyp](http://www.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/) datasets. Its performance exceeds the classical UNet model on all these three datasets. 

Data format should be like:
```
+ Contour_aware_unet
    + dataset_name
        + image
            + patient1.jpg
            + patient2.jpg
            + patient3.jpg
            ...
        + mask
            + patient1.png
            + patient2.png
            + patient3.png
            ...
```

## Dependencies
```
torch == 1.6.0
torchvision == 0.7.0
```
Multi-GPU is also recommanded for accelarating the training process.

## Train and Predict
The training and prediction modules are wrapped in the same file `train_with_border.py`, in the form of an ipy-notebook, so if you run the script in VS Code or other suitable IDEs, the whole procedure can be split into successive blocks.

Or you can directively run `python train_with_border.py` to execute the whole script at once.