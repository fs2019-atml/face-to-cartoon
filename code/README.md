# Code notes
Our contributions are declared using the following comment:
`#### GROUP5 code ####` and `#### END of code ####`.

Most are found under
* `data/face_dataset.py` (our own data handling and augmentation incorporating landmarks)
* `model/cycle_gan_model.py` (adaptation to handle own data types and conditional networks)
* `model/networks.py` (the conditional architecture, the landmark network (LDNet), Own loss functions)
* `landmarks/*` (matlab script to label faces with landmarks)
* `cam.py` (demo script to use our model of the commandline)

## Acknowledgments
Our code is a fork of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
