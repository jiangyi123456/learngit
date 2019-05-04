import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

plotdata={'batchsize':[],'loss':[]}

train_x=np.linspace(-1,1,100)
train_y=2*train_x+np.random.randn(100)*0.3

tf.reset_default_graph()#初始化化图

x=tf.placeholder('float')
y=tf.placeholder('float')

w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bise')

z=tf.multiply(x,w)+b
tf.summary.histogram('z',z ) #直方图显示

cost=tf.reduce_mean(tf.square(y-z))
tf.summary.scalar('cost',cost) #标量显示

learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init=tf.global_variables_initializer()
train_epoch=20
display_epoch=2
saver=tf.train.Saver(max_to_keep=1)
savedir='log/'
with tf.Session() as sess:
    sess.run(init)
    mergar_summery_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter('log/with_summaries',sess.graph)
    for epoch in range(train_epoch):
        for (x1,y1) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict=({x:x1,y:y1}))
            summary_str=sess.run(mergar_summery_op,feed_dict=({x:x1,y:y1}))
            summary_writer.add_summary(summary_str,epoch)
        if epoch % display_epoch==0:
            loss=sess.run(cost,feed_dict=({x:train_x,y:train_y}))
            print('epoch:',epoch+1,'cost', loss, sess.run(w), sess.run(b))
            plotdata['batchsize'].append(epoch)
            plotdata['loss'].append(loss)
            saver.save(sess,savedir+'linermodel.cpkt',global_step=epoch)
    plt.plot(train_x, train_y, 'ro')
    plt.plot(train_x,train_x*sess.run(w)+sess.run(b))
    plt.legend()
    plt.show()
    print('x=0.2:',sess.run(z,feed_dict=({x:0.2})))
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+'linermodel.cpkt-'+ str(18)) # 导出保存的模型
    print(sess2.run(z,feed_dict=({x:0.2})))
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess3,ckpt.model_checkpoint_path)

        print(sess3.run(z,feed_dict=({x:0.2})))
with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer())
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt!=None:
        saver.restore(sess4,kpt)
        print(sess4.run(z,feed_dict=({x:0.2})))