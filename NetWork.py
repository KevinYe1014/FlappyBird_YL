
import tensorflow as tf
import cv2
import numpy as np
from FlappyBird.flappy_bird import GameState
from collections import deque
import random


ACTIONS=2 ##number of valid actions
GAME='BIRD' ##the name of the game being played for log files
INITIAL_EPSION=0.0001 ##starting value of epsilon
FINAL_EPSILON=0.0001 ##final value of epsilon
GAMMA=0.99 ##decay rate of past obervations
OBSERVE=100000 ##timesteps to observe before traing
EXPLORE=2000000 #frames over which to anneal epsilon
REPLAY_MEMORY=50000 ##number of previous transitions to remember
BATCH=32 ##size of minibatch
FRAME_PER_ACTION=1




def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.01,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w,stride):
    return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],padding='SAME')

def createNetWork():
    ##第一层卷积层
    w_conv1=weight_variable([8,8,4,32])
    b_conv1=bias_variable([32])

    w_conv2=weight_variable([4,4,32,64])
    b_conv2=bias_variable([64])

    w_conv3=weight_variable([3,3,64,64])
    b_conv3=bias_variable([64])

    w_fc1=weight_variable([1600,512])
    b_fc1=bias_variable([512])

    w_fc2=weight_variable([512,ACTIONS])
    b_fc2=bias_variable([ACTIONS])
    ##输入层
    S=tf.placeholder('float',[None,80,80,4])
    h_conv1=tf.nn.relu(conv2d(S,w_conv1,4)+b_conv1)
    h_pool1=max_pool_2(h_conv1)

    h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2,2)+b_conv2)

    h_conv3=tf.nn.relu(conv2d(h_conv2,w_conv3,1)+b_conv3)
    ##拉平
    h_conv3_flat=tf.reshape(b_conv3,[-1,1600])
    ##全连接层
    h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)
    ##输出层
    readout=tf.matmul(h_fc1,w_fc2)+b_fc2

    return S,readout,h_fc1

def trainNetWork(s,readout,h_fc1,sess):
    #定义损失函数
    a=tf.placeholder('float',[None,ACTIONS])
    y=tf.placeholder('float',[None])

    readout_action=tf.reduce_sum(tf.multiply(readout,a),reduction_indices=1)
    cost=tf.reduce_mean(tf.square(y-readout_action))
    train_step=tf.train.AdamOptimizer(1e-6).minimize(cost)

    ##定义游戏状态收集emulator
    game_state=GameState()
    ##store the previous observations in replay memory
    D=deque()

    ##printing
    a_file=open('logs_'+GAME+'/readout.txt','w')
    h_file=open('logs_'+GAME+'/hidden.txt','w')

    ##get the first state by doing nothing and preprocess the image to 80*80*4
    do_nothing=np.zeros(ACTIONS)
    do_nothing[0]=1

    x_t,r_0,terminal=game_state.frame_step(do_nothing)

    #####图像转为灰度图 并且二值化
    x_t=cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_RGB2GRAY)
    ret,x_t=cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t=np.stack((x_t,x_t,x_t,x_t),axis=2)

    ##saving and loading networks
    saver=tf.train.Saver()

    sess.run(tf.initialize_all_variables())
    checkpoint=tf.train.get_checkpoint_state('saved_networks')

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess,checkpoint.model_checkpoint_path)
        print("Successfully loaded:",checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    ##start training
    epsilon=INITIAL_EPSION
    t=0
    while 'flappy bird' !='angry bird':
        '''choose an action epsilon greedily'''
        readout_t=readout.eval(feed_dict={s:[s_t]})[0]
        a_t=np.zeros(ACTIONS)

        action_index=0
        if t%FRAME_PER_ACTION==0:
            if random.random()<=epsilon:
                print("-------------Random Action-------------")
                action_index=random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)]=1
            else:
                action_index=np.argmax(readout_t)
                a_t[action_index]=1
        else:
            a_t[0]=1 ##do nothing [0,1]:do nothing [1,0]:click


        ##scale down epsilon
        if epsilon>FINAL_EPSILON and t>OBSERVE:
            epsilon-=(INITIAL_EPSION-FINAL_EPSILON)/EXPLORE

        ##run the selected action and observe next state and reward
        x_t1_colored,r_t,terminal=game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        ##store the transition in D
        D.append(s_t,a_t,r_t,s_t1,terminal)
        if len(D)>REPLAY_MEMORY:
            D.popleft()

        ##only train if done obersving
        if t>OBSERVE:
            #samples a minibatch to brain on
            minibatch=random.sample(D,BATCH)

            ##get the batch variables
            s_j_batch=[d[0] for d in minibatch]
            a_batch=[d[1] for d in minibatch]
            r_batch=[d[2] for d in minibatch]
            s_j1_batch=[d[3] for d in minibatch]

            y_batch=[]
            readou_j1_batch=readout.eval(feed_dict={s:s_j1_batch})
            for i in range(len(minibatch)):
                terminal=minibatch[i][4]
                ##if terminal , only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i]+GAMMA+np.max(readou_j1_batch[i]))

            ##perform gradient step
            train_step.run(feed_dict={y:y_batch,a:a_batch,s:s_j_batch})

        ##update the old values
        s_t=s_t1
        t+=1

        ##save progress every 10000 iterations
        if t% 10000==0:
            saver.save(sess,'saved_network/'+GAME+'-dqn',global_step=t)

        ##print info
        state=''
        if t<=OBSERVE:
            state='observe'
        elif t>OBSERVE and t<=OBSERVE+EXPLORE:
            state='explore'
        else:
            state='train'

        print('TIMESTEP',t,'/STATE',state,'/EPSILON',epsilon,'/ACTION',action_index,'/REWARD',r_t
              ,'/Q_MAX%e'%np.max(readout_t))
        ##write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
def playGame():
    sess=tf.InteractiveSession()
    s,readout,h_fc1=createNetWork()
    trainNetWork(s,readout,h_fc1,sess)
def main():
    playGame()

if __name__=='__main__':
    main()



































