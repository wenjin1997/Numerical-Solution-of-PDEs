import matplotlib.pyplot as plt #在程序开头引用这个画图的包

# begin drawing
plt.title('Result') # 这里可以修改图的标题
# 其中一条曲线，自变量为t，因变量为u_eluer，加上颜色和标签，自变量和因变量根据自己程序中要画的作修改
plt.plot(t, u_euler, color='green', label='euler') 
plt.plot(t, u, color='blue', label='implict euler') # 第二条曲线，颜色为蓝色
plt.plot(t, u_true, color='red', label='exact') # 第三条曲线，颜色为红色
plt.legend() # show the legend 

plt.xlabel('t') #设置整个图像的横坐标的名称
plt.ylabel('u') # 设置整个图像的纵坐标的名称
plt.show() # 展示图像