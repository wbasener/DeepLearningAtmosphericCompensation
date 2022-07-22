# DeepLearningAtmosphericCompensation
This is an autoencoder that learns the radiative atmosphere transfer to convert hyperspectral data from radiance to reflectance.

We treat the atmospheric affects like 'noise' in a common denoising autoencoder, and learn this noise through trainning.  It does not as implemented work sufficiently yet for atmostpheric compensation of hyperspectral images, but we expect that the idea is sound and potentially extraorinarily valuable is successful, and as with most deep learning models there is likely improvements to the architecture that can cause leaps forward in accuracy.  

<img src="https://user-images.githubusercontent.com/51686251/180453646-198c14dc-73a7-400e-9ddd-8281ea568735.png" style="width:200px;margin-left:15px;float:right">

![image](https://user-images.githubusercontent.com/51686251/180453866-7a164f7f-6440-4028-b79b-c10ee607479d.png)

![image](https://user-images.githubusercontent.com/51686251/180454007-815c3277-0924-4123-bc2f-9fd23c244ab1.png)


Our paper on this is archived at: https://arxiv.org/abs/2207.10650
