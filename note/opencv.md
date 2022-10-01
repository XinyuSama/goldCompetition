# 1.图像的基本操作

计算机图像中的每个像素点是0-255，亮度

基本的图像有三个通道R.G.B->红绿蓝![image-20220929161756006](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929161756006.png)

在计算机眼中，图像是以数字矩阵的形式存储的，一张图片被分成了若干个小方格，但是小方格还没有足够小，随意取出一个方格放大，又有许多更小的方格，这个小的不能再小的方格叫做一个像素点，像素点有对应的值，在计算机中像素点的值在0，255之间，数字越大表示这个像素点越亮，一张彩色的图片有R、G、B三个颜色通道，每个通道上像素点的值代表该通道上的亮度。对于灰度图也就是黑白图来说他只有一个颜色通道，一张图片的维度可以表示为【h,w,c】其中h代表高度方向的像素点个数，w代表宽度方向的像素点个数，c代表颜色通道数。

![img](https://pic2.zhimg.com/v2-f4ce00607fdff73c587662daec3c95a9_r.jpg)

一张图片的颜色是由RGB三个通道构成, 可以把一张图片上的每一个像素点看成一个对象, 这个对象又由RGB三种颜色叠加, 即用一个一维数组表示,假如我们有一张 m * n 个像素点的图片, 那么每一行有 n 个像素, 即每一行有 n 个一维数组, 即这一行是一个二维数组, 那一张图片又有 m 行, 那么我们就得到了 m 个二维数组, 这m 个二维数组构成了一个三维数组。<font color=red>（即每个最内层的数组有三个元素，代表着RGB三个通道的灰度值。第二层和第三层则负责遍历整个行和列）</font>

## 1.1 数据读取-图像

- `cv2.IMRREAD_COLOR`:彩色图像 三通道 三维
- `cv2.IMREAD_GRAYSCALE`:灰度图像 一通道

```python
import cv2 #opencv读取的格式是BGR
import matplotlib.pyplot as plt # 最好用opencv的api进行展示
import numpy as np

%mapplotlib inline # notebook #魔法语句 直接不用show方法就可以展示图像 

img = cv2.imread('cat.jpg')
# 读出来是一个ndarray,dtype:uint = 8 (0-255)的一个三维数组->[h,w,c]即[hight,width,color] 


array([
        [[15, 197, 44],
        [55, 197, 244],
        [156, 198, 25],
        ...,
        [11, 178, 35],
        [105, 172, 229],
        [02, 169, 226]],
 
       [[150, 192, 39],
        [11, 193, 240],
        [152, 194, 241],
        ...,
        [118, 47, 204],
        [118, 47, 204],
        [118, 47, 204]]
        ...], dtype=uint8)

# 图像的显示，也可以创建多个窗口
cv2.imshow('image',img)
# 等待时间，毫秒级，0表示任意键终止
cv2.waitKey(0)
cv2.destroyAllWindows()

# 封装成一个函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(o)
    cv2.destroyAllWindows()

print(img.shape) # (414,500,3) 即（h,w,c）

# 读的时候直接读为灰度图像
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
```

## 1.2 数据读取-视频

- cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0，1
- 如果是视频文件，直接指定好路径即可
- 一帧就是一张图，视频就是一帧一帧的图像连贯起来的
- 处理视频就是处理每一帧图

```python
vc = cv2.VideoCapture("test.mp4")

# 检查是否打开正确
if vc.isOpened():
    # 读取每一帧 open 为读取成功或失败 frame即为一帧图像的数据
    open,frame = vc.read()
else:
    open = False
while open:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        # 转换为灰度
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        # 读完每一帧的等待时间
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
    
```

# 2.ROI区域

## 2.1 截取部分图像数据

```python
img = cv2.imread('cat.jpg')

cat = img[0:50,0:200] # h w的截取

cv_show('cat',cat)
```

## 2.2 颜色通道提取

```python
# 颜色通道提取
b,g,r = cv2.split(img)
# 合成
img = cv2.merge((b,g,r))

# 只保留R
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0
cv_show('R',cur_img)
# 只保留B......
```



# 3.边界填充

```python
# opencv 4.5版本不存在copyMakeBorder函数
top_size,bottom_size,left_size,right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType = cv2.BODER_REPLICATE)
```

borderType：

1. REPLICATE：复制法，也就是复制最边缘像素
2. REFLECT:反射法，对感兴趣图像中的像素在两边进行复制 例如：fedcba|abcdefgh|hgfedcb
3. REFLECT_101:反射法，以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
4. WRAP:外包装法cdefgh|abcdefgh|abcdefg
5. CONSTANT:常量法，常量值填充

# 4.数值计算

```python
# 读取图片
image = cv2.imread("./lena.png")
image2 = cv2.imread("./lena.png")
image2 = image+10
print(image2[:5,:,0]) # 0代表平铺
```

在numpy中如果加到大于256，比如是294，就会拿294-256=38

但是如果使用opencv提供的`cv2.add()`就可以如果大于256，就取256

## 4.1 图像融合

`img_cat`+`img_dog`

但是注意,两个图像的形状shape不同，会报错，就需要使用`cv2.resize()`函数更改图像形状

```python
img_dog = cv2.resize(img_dog,(500,414)) # 注意 500是w,414是h 别弄反了
img_dog = cv2.resize(img_dog,(0,0)，fx=3,fy=1) # w是原来的3倍，h是原来的1倍
```

cv2的融合：

```python
res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0)
# 即 img_cat * 0.4 + img_dog * 0.6 + b 
# b为偏置项 亮度
```

# 5.图像阈值

每个像素的通道值大于小于某个值，怎么样怎么样

![image-20220929184310618](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929184310618.png)

_INV即反转方法

```python
import cv2
import pandas as plt

image = cv2.imread("./lena.png",cv2.IMREAD_GRAYSCALE)

ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)

titles = ["Original Image","BINARY","BINARY_INY","TRUNC","TOZERO","TOZERO_INV"]
images = [image,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```

# 6.图像平滑处理

## 6.1 均值滤波

简单的平均卷积操作

![image-20220929213144465](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929213144465.png)

```python
blur = cv2.blur(img,(3,3)) # 3*3卷积核的大小

cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 方框滤波
# 基本和均值一样，做法也差不多，但是没有除以操作，可以选择归一化 
box = cv2.boxFilter(img,-1,(3,3),normalize = True ) # -1指自动计算 基本上不用改 normalize：做不做归一化 容易越界

# 越界 超过255，就直接取255

cv2.imshow('box',box)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 6.2 高斯滤波

权重矩阵 重要程度有远近操作 更重视中间的，越近越大，越远越小

![image-20220929220413185](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929220413185.png)

```python
# 高斯模糊的卷积核里的数值是满足高斯分布，更重视中间的

# 核就是一个矩阵

aussian = cv2.GaussianBlur(img,(5,5),1)

cv2.imshow('aussian',aussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 6.3 中值滤波

```python
# 使用中值代替

median = cv2.medianBlur(img,5)

cv2.imshow('median',median)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 6.4 展示所有

```python
res = np.hstack((blur,aussian,median))
print(res)
cv2.imshow('vs',res )
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 7.形态学操作

拿一个核来对周围的点进行计算

## 7.1 腐蚀操作

去毛刺，有价值的数据越来越少

```python
kernel = np.ones((5,5),np.uint8) # 定义一个5*5的核 类型为uint8 即0-255之间
erosion = cv2.erode(img,kernel,iterations =2) # iterations为腐蚀迭代次数 

cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220929224526265](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929224526265.png)

如果一个点周围的颜色都和这个点颜色不太相同，就把它腐蚀掉变成相同。

## 7.2 膨胀操作

和腐蚀相反,互为逆运算

```python
kernel = np.ones((5,5),np.uint8) # 定义一个5*5的核 类型为uint8 即0-255之间
dige_dilate = cv2.dilate(img,kernel,iterations =2) # 和腐蚀参数一样

cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 7.3 开闭运算

开闭运算就是把腐蚀，膨胀操作连贯起来

- 开 ：先腐蚀，再膨胀
- 闭 ：先膨胀，后腐蚀，把开运算逆过来

开：

```python
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) #卷积核
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) # cv2.MORPH_OPEN

cv2.imshow('opening',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

闭：

```python
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) #卷积核
opening = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) # cv2.MORPH_CLOSE

cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 7.4 梯度运算

梯度 = 膨胀 - 腐蚀 =>>>>>得到一个边框信息 这个相减操作就是一个梯度

```python
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel) # cv2.MORPH_GRADIENT

cv2.imshow('gradient',gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 7.5 礼帽和黑帽

- 礼貌 = 原始输入 - 开运算结果（先腐蚀后膨胀）
  - 原来有刺 - 不带刺 = 刺
- 黑帽 = 闭运算（先膨胀后腐蚀） - 原始输入
  - 刺有点多 - 原来带刺 = 剩下点刺

礼帽：

```python
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) #卷积核
opening = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel) # cv2.MORPH_TOPHAT

cv2.imshow('tophat',tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

黑帽：

```python
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) #卷积核
opening = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel) # cv2.MORPH_BLACKHAT

cv2.imshow('tophat',tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 8.图像梯度

数值不同才会产生梯度 ， 一般用在边缘检测

## 8.1 Sobel算子

和前面一样需要定义一个核来进行计算，但要考虑上下左右的差异，右减左，下减上

![image-20220929232346464](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929232346464.png)

Gx表示水平方向上下左右的的梯度（差异值），Gy同理

为什么会有 -1 -2 +1 +2 ?  因为离的越近值越大，类似高斯分布



`dst = cv2.Sobel(src,ddepth,dx,dy,ksize)`

- ddepth:图像的深度
- dx和dy分别表示水平和竖直方向
- ksize是Sobel算子的大小

![image-20220929234029674](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929234029674.png)

算x:右边减去左边

```python
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize =3 )
# cv2.CV_64F opencv会把负值截断为0，这样写就把位数增多，1表示true,0为false ,即算dx
sobelx = cv2.convertScaleAbs(sobelx) # 算负数的绝对值
# 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
cv2.imshow('sobelx',sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220929234723712](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929234723712.png)

算y:下面减上面

```python
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =3 )
# cv2.CV_64F opencv会把负值截断为0，这样写就把位数增多，1表示true,0为false ,即算dy
sobely = cv2.convertScaleAbs(sobely) # 算负数的绝对值
# 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
cv2.imshow('sobely',sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220929234732015](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929234732015.png)

求和 x+y 融合

```python
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow('sobelxy',sobelxy)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220929234758678](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929234758678.png)





为什么不直接把dx和dy都直接设置为1？效果不太好！

## 8.2 Scharr算子和laplacian算子



![image-20220929234928737](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929234928737.png)

### 8.2.1 Scharr

对结果的差异更明显

### 8.2.2 laplacian

对变化更敏感，对噪音点敏感，不是下面上面左右，是边缘点，不常用

![image-20220929235313135](F:\Project\43.goldCompetition\note\opencv.assets\image-20220929235313135.png)

# 9.Canny边缘检测流程

1. 使用高斯滤波器，以平滑图像，滤除噪声
2. 计算图像中每个像素点的梯度强度和方向
3. 应用非极大值（Non-Maximum Supperession）抑制，以消除边缘检测带来的杂散响应
4. 应用双阈值（Double-Threshold)检测来确定真实的和潜在的边缘
5. 通过抑制孤立的弱边缘最终完成边缘检测

## 9.1 高斯滤波器

![image-20220930095125699](F:\Project\43.goldCompetition\note\opencv.assets\image-20220930095125699.png)

## 9.2 计算梯度强度和方向

采用的Sobel算子

算Gx和Gy

![image-20220930095209220](F:\Project\43.goldCompetition\note\opencv.assets\image-20220930095209220.png)

## 9.3 非极大值抑制

为什么要非极大值抑制？把梯度强度最大的保留下来，把最明显的保留下来

### 法1(复杂):

![image-20220930095600327](F:\Project\43.goldCompetition\note\opencv.assets\image-20220930095600327.png)

### 法2：

![image-20220930095707238](F:\Project\43.goldCompetition\note\opencv.assets\image-20220930095707238.png)

## 9.4 双阈值检测

梯度值 > maxVal : 则处理为边界

minVal < 梯度值 < maxVal : 连有边界则保留，否则舍弃

梯度值 < minVal : 则舍弃

![image-20220930192447265](F:\Project\43.goldCompetition\note\opencv.assets\image-20220930192447265.png)

AC保留，B舍弃，因为C连有边界。

如果minVal太小，就说明要求较小，条件放松，

```python
img = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img,120,250)
v2 = cv2.Canny(img,50,100)

res = np.hstack((v1,v2)) # 合并
```

# 10.图像金字塔定义

## 10.1 图像金字塔

- 高斯金字塔
- 拉普拉斯金字塔

![image-20221001125031725](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001125031725.png)

做特征提取，每一层进行特征提取。

### 10.1.1 高斯金字塔

- 向下采样（缩小）

![image-20221001125236866](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001125236866.png)

1/16是做归一化

去除偶数行和列，面积就变成原来1/4

```python
down = cv2.pyrDown(img)
```



* 向上采样（放大）

![image-20221001125620089](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001125620089.png)



```python
up = cv2.pyrUp(img)
```

### 10.1.2 拉普拉斯金字塔

![image-20221001125925151](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001125925151.png)

原理：

![image-20221001130016190](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001130016190.png)

# 11.图像轮廓

![image-20221001130209642](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001130209642.png)

轮廓方法：

![image-20221001130537250](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001130537250.png)



## 11.1 绘制轮廓

为了更高的准确率，使用二值图像

```python
img = cv2.imread('contours,png')
# 转换为灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值处理
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# 轮廓提取
binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #返回 原图，轮廓信息，层级

# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy 要不原图会变
draw_img = img.copy()
res = cv2.drawContours(img,contours,-1,(0,0,255),2) # -1是指画第几个轮廓 -1值所有 （0,0,255）BGR , 2指线条的宽度

```

## 11.2 轮廓特征

```python
cnt = contours[0] # 取第一个轮廓信息
# 面积
cv2.contourArea(cnt)
# 周长，True表示闭合的
cv2.arcLength(cnt,True)
```

## 11.3 轮廓近似

![image-20221001133531035](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001133531035.png)

无限计算，类似二分进行判断曲线是否可以用直线代替

```python
img = cv2.imread('contours,png')
# 转换为灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值处理
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# 轮廓提取
binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #返回 原图，轮廓信息，层级

# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy 要不原图会变
draw_img = img.copy()
res = cv2.drawContours(img,contours,-1,(0,0,255),2) # -1是指画第几个轮廓 -1值所有 （0,0,255）BGR , 2指线条的宽度

epsilon = 0.1*cv2.arcLength(cnt,True) # 0.1倍周长 来做阈值 倍数越小 越精细
approx = cv2.approxPolyDP(cnt,epsilon,True) # 做近似

draw_img = img.copy()
res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
```

## 11.4 边界矩形

```python
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) 
```

### 11.4.1 轮廓面积与边界矩形比

```python
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area) / rect_area # 轮廓面积与边界矩形比
```

## 11.5 外接圆

```python
(x,y).radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)
```

# 12.模板匹配方法

拿到一张原始图像，和另一张图像进行匹配。

![image-20221001135623478](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001135623478.png)

模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板（图像被模板覆盖的地方）的差别程度，这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入一个矩阵里，作为结果输出，假如原图像是AxB大小，而模板是axb大小，则输出结果的矩阵是(A-a+1)x(B-b+1)



![image-20221001141930633](F:\Project\43.goldCompetition\note\opencv.assets\image-20221001141930633.png)

```python
img = cv2.imread('lena.png',0)
template = cv2.imread('face.jpg',0)
h,w = template.shape[:2]
res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF) # TM_SQDIFF也可以用1代替

# 主要关注最小值  min_loc 因为有宽高

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) # 定位
```

案例：

```python

methods = ['cv2.TM_CCOEFF'.......]

for meth in methods:
    img2 = img.copy()
    
    # 匹配方法的真值
    method = eval(meth)
    
    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) # 定位
    
    #如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED,取最小值
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] +h )
    
    # 画矩形
    cv2.rectangle(img2,top_left,bottom_right,255,2)
    
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    plt.show()
```

没有归一化的效果都比较差 后缀带`_NORMED`的都是归一化

## 12.1 匹配多个对象

```python
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg' , 0)
h,w = template.shape[:2]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) # res返回每个窗口的值
threshold = 0.8
# 取匹配程度大于80%的坐标 就画出来
loc = np.where(res>=threshold)
for pt in zip(*loc[::-1]): # *号代表可选参数
    bottom_right = (pt[0] + w , pt[1] + h )
    cv2.rectangle(img_rgb,pt,bottom_right,(0,0,255),2)
cv2.imshow('img_rgb' , img_rgb)
cv2.waitKey(0)
```

# 13.直方图定义。。
