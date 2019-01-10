# Style_transfer_pyProject
Just a small exercise with Style-Transfer and Python UI


一、	简介
本项目的功能为将自选图片进行风格转换。项目参考了"Image Style Transfer Using Convolutional Neural Networks" (Gatys et al., CVPR 2015)的方法，利用一个轻量级而高效的网络SqueezeNet作为特征提取器，来提取原始图像的结构信息以及画作的风格信息，定义对应结构内容和风格的损失函数，运行梯度下降，生成一张新的图片。本项目的神经网络框架利用Pytorch实现。用户可以自选原始图片以及名家画作，对原图进行对应的风格转换，显示生成的图片并保存到本地。


二、	用到的库
1.	Numpy
2.	torch
3.	torchvision
4.	PIL
5.	Tkinter


三、	使用说明
1.	初始界面
2.	点击“选择原图”按钮选择原图
3.	点击“选择风格图片”按钮选择画作
4.	点击“转换！”按钮开始转换，此时需要等待几分钟（真的不是卡住）
5． 点击“另存为”可以将图片保存到本地


Reference
[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. Computer Vision & Pattern Recognition.
[2] Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size. arXiv preprint arXiv:1602.07360.
