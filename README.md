# cuda_practice
cuda编程简单实践

分别使用cpu和gpu完成了二维矩阵的加法和乘法运算。二维矩阵的大小为1024*1024。

在i7 9750h和RTX 1660Ti上的运行结果如下所示：
|运算类型|运算工具|耗时|
|:---:|:---:|:---:|
|加法|cpu|5ms|
|加法|gpu|3ms|
|乘法|cpu|9007ms|
|乘法|gpu|80ms|

**使用方法**
在linux上运行如下命令：
* git clone https://github.com/buddhisant/cuda_practice.git
* cd cuda_practice
* nvcc multiply.cu -o multiply
* ./multiply
