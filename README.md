## [SaGAAN (For hyperspectral sample generation)](https://www.mdpi.com/2072-4292/12/5/843)

SaGAAN aims to generate realistic hyperspectral profiles by feeding a small amount of labeled samples. It is developed to alleviate the problem of sample shortage when performing hyperspectral image classification. There are two contributions in this work, being 1) self-attention is introduced to waive unwanted noises, and 2) domain adaptation reinforced the similarity of generated samples. The SaGAAN not only helpful in hyperspectral data analysis domain, it is possible to be useful in time-series data processing. Please cite this work if you find it interesting.

["Sample Generation with Self-Attention Generative Adversarial Adaptation Network (SaGAAN) for Hyperspectral Image Classification"](https://www.mdpi.com/2072-4292/12/5/843)

**Note:**
To use the SaGAAN code, you may want to make sure the necessary environment already satisfied. The related modules or packages are:
- tensorflow
- matplotlib
- scipy
- etc...

**How to use:**
You should aware this code is in its prototype format, you may need to revise parameters according to your datasets or tasks. Here is the general overview of this project.

### Run "SaGAAN.py" to generate samples
You should make sure all files are in right positions.
1. check lines 168, 227-228, make sure all training files are ready;
2. check generator and discriminator have the correct setting (here is Pavia dataset config)
3. check line 384, make sure the output samples is in right file

At last you should plot the generated samples and compare them with the original ones, like this 
![SaGAAN](https://www.mdpi.com/remotesensing/remotesensing-12-00843/article_deploy/html/images/remotesensing-12-00843-g002-550.jpg)  
