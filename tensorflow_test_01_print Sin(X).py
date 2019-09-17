#題目:請使用tensorflow建立一個神經網路，模擬出Sin(x)，並使用matplotlib打印出訓練過程
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Sin圖的公式
x = np.arange(0,4*np.pi,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()

#placeholder(從神經網路外干涉)
time_placeholder = tf.placeholder(tf.float32, [None])
value_placeholder = tf.placeholder(tf.float32, [None])

#change shape to [batch_size, 1]
time = tf.expand_dims(time_placeholder,axis=1)
value = tf.expand_dims(value_placeholder,axis=1)

#建立Full connect layer
x1 = tf.layers.dense(time, 100, activation=tf.nn.tanh)
x2 = tf.layers.dense(x1, 25, activation=tf.nn.tanh)
x3 = tf.layers.dense(x2, 50, activation=tf.nn.tanh)
x4 = tf.layers.dense(x3, 25, activation=tf.nn.tanh)
predict_value = tf.layers.dense(x4, 1, activation=tf.nn.tanh)

#loss function MAE
loss = tf.losses.mean_squared_error(value, predict_value)

#minimize
optimizer = tf.train.AdamOptimizer(8e-4)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        output_time, output_value, output_predict, _ = sess.run(
                [time, value, predict_value, train_op],
                 feed_dict={time_placeholder: x, value_placeholder: y})
        
        #print it
        #用取餘數的方式來調整epoch數量
        if epoch % 1000 == 0:
            plt.figure()
            plt.plot(output_time, output_value)
            plt.plot(output_time, output_predict)
            plt.show()
