# DeepLearningAtmosphericCompensation
This is an autoencoder that learns the radiative atmosphere transfer to convert hyperspectral data from radiance to reflectance.

We treat the atmospheric affects like 'noise' in a common denoising autoencoder, and learn this noise through trainning.  It does not as implemented work sufficiently yet for atmostpheric compensation of hyperspectral images, but we expect that the idea is sound and potentially extraorinarily valuable is successful, and as with most deep learning models there is likely improvements to the architecture that can cause leaps forward in accuracy.  

Our paper on this is archived at: https://arxiv.org/abs/2207.10650
