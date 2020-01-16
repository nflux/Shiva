import os
import csv
import pandas as pd
import numpy as np
import datetime
import time

metrics = ["Reward/Per_Step", "Reward/Per_Episode"]
env = ["CartPole-v0","3DBall", "Basic", "MountainCarContinuous-v0"]
algType = ["DQNAlgorithm","PPOAlgorithm", "ContinuousDDPGAlgorithm"]
output = []
rate = 5


#====================================DQN Cartpole Benchmak=======================
# For use in DQN.ini 
def testRoutineDQN():
    output = []

    os.chdir("..")
    try: 
        pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[0]+" "+env[0] +" "+ ""+'.csv', sep=',',header=None)
    except: 
        print("PPO FAILED")
        os.chdir("benchmarkCode")
        openReadmeFile("       |Cartpole v0|N/A|       **DQN**     |       FAILED     |")
        return
    df=pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[0]+" "+env[0] +" "+ ""+'.csv', sep=',',header=None)
    
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    length = len(res[1])
    checkRate = (length/rate)
    # print(res[1][999])
    i = 0
    while (i < length):
        i =  int(i+ round(checkRate,10))
        output.append(np.average(res[1][0:i]))
 
    os.chdir("benchmarkCode")
    k = 0
    write = ""
    for string in output:
        k = k+ round(checkRate,10)
        write = write + " " +str(string) +": 0-"+ str(int( k)) +" Steps, <br>            "
    openReadmeFile("       |Cartpole v0|"+str(length)+"|       **DQN**     |       "+  write+"   |"  + maxVolume(res[1]+"|"))
  
#====================================DQN Unity Basic Benchmak=======================
# For use in DQN-Unity-Basic.ini
def testRoutineDQNUnityBasic():
    output = []

    os.chdir("..")
    try: 
        pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[0]+" "+env[2] +" "+ ""+'.csv', sep=',',header=None)
    except: 
        print("PPO FAILED")
        os.chdir("benchmarkCode")
        openReadmeFile("       |Unity Basic|N/A|       **DQN**     |       FAILED     |")
        return
    df=pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[0]+" "+env[2] +" "+ ""+'.csv', sep=',',header=None)
    
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    length = len(res[1])
    checkRate = (length/rate)
    # print(res[1][999])
    i = 0
    while (i < length):
        i =  int(i+ round(checkRate,10))
        output.append(np.average(res[1][0:i]))
 
    os.chdir("benchmarkCode")
    k = 0
    write = ""
    for string in output:
        k = k+ round(checkRate,10)
        write = write + " " +str(string) +": 0-"+ str(int( k)) +" Steps, <br>            "
    openReadmeFile("       |Unity Basic|"+str(length)+"|       **DQN**     |       "+  write+"   |" + maxVolume(res[1]+"|")
  


#====================================PPO Cartpole Benchmak=======================
# For use in PPO.ini
def testRoutinePPO():
    output = []
    # file = open("Benchmark/"+str(metrics[1])+" " +env[0]+" "+algType[0] +" "+ ""+'.csv', 'w+')
    # return file
    os.chdir("..")
    print (os.path.abspath(os.curdir))
    try: 
        pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[1]+" "+env[0] +" "+ ""+'.csv', sep=',',header=None)
    except: 
        print("DQN FAILED")
        os.chdir("benchmarkCode")
        openReadmeFile("       |Cartpole v0| N/A |       **PPO**     |       FAILED     |")
        return
    df=pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[1]+" "+env[0] +" "+ ""+'.csv', sep=',',header=None)
    
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    length = len(res[1])
    checkRate = length/rate
    i = 0
    
    while (i < length):
        i =  int(i+ round(checkRate,10))
        output.append(np.average(res[1][0:i]))
 
    os.chdir("benchmarkCode")
    k = 0
    write = ""
    for string in output:
        k = k+ round(checkRate,10)
        write = write + " " +str(string) +": 0-"+ str(int( k)) +" Steps, <br>            "
    openReadmeFile("       |Cartpole v0|"+str(length)+"|       **PPO**     |       "+  write+"   |" + maxVolume(res[1]+"|")




#====================================DDPG-3DBALL Benchmark=======================
#For Use in DDPG-3DBall.ini
def testRoutineDDPG3DBALL():
    output = []
    # file = open("Benchmark/"+str(metrics[1])+" " +env[0]+" "+algType[0] +" "+ ""+'.csv', 'w+')
    # return file
    os.chdir("..")
    print (os.path.abspath(os.curdir))
    try: 
        pd.read_csv("Benchmark/"+str(metrics[0])+" " +algType[2]+" "+env[1] +" "+ ""+'.csv', sep=',',header=None)
    except: 
        print("DDPG FAILED")
        os.chdir("benchmarkCode")
        openReadmeFile("       |Unity 3D-BALL|N/A|       **DDPG**     |       FAILED     |")
        return
    df=pd.read_csv("Benchmark/"+str(metrics[0])+" " +algType[2]+" "+env[1] +" "+ ""+'.csv', sep=',',header=None)
    
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    length = len(res[1])
    checkRate = length/rate
    i = 0
    
    while (i < length):
        i =  int(i+ round(checkRate,10))
        output.append(np.average(res[1][0:i]))
 
    os.chdir("benchmarkCode")
    k = 0
    write = ""
    for string in output:
        k = k+ round(checkRate,10)
        write = write + " " +str(string) +": 0-"+ str(int( k)) +" Steps, <br>            "
    openReadmeFile("       |Unity 3D-BALL|"+str(length)+"|       **DDPG**     |       "+  write+"   |" + maxVolume(res[1]+"|")


#====================================DDPG-Mountain Car Benchmark=======================
#For Use in DDPG-3DBall.ini
def testRoutineDDPGMountainCar():
    output = []
    # file = open("Benchmark/"+str(metrics[1])+" " +env[0]+" "+algType[0] +" "+ ""+'.csv', 'w+')
    # return file
    os.chdir("..")
    print (os.path.abspath(os.curdir))
    try: 
        pd.read_csv("Benchmark/"+str(metrics[0])+" " +algType[2]+" "+env[3] +" "+ ""+'.csv', sep=',',header=None)
    except: 
        print("DDPG FAILED")
        os.chdir("benchmarkCode")
        openReadmeFile("       |MountainCarContinuous-v0|N/A|       **DDPG**     |       FAILED     |")
        return
    df=pd.read_csv("Benchmark/"+str(metrics[0])+" " +algType[2]+" "+env[3] +" "+ ""+'.csv', sep=',',header=None)
    
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    length = len(res[1])
    checkRate = length/rate
    i = 0
    
    while (i < length):
        i =  int(i+ round(checkRate,10))
        output.append(np.average(res[1][0:i]))
 
    os.chdir("benchmarkCode")
    k = 0
    write = ""
    for string in output:
        k = k+ round(checkRate,10)
        write = write + " " +str(string) +": 0-"+ str(int( k)) +" Steps, <br>            "
    openReadmeFile("       |MountainCarContinuous-v0|"+str(length)+"|       **DDPG**     |       "+  write+"   |" + maxVolume(res[1]+"|")



def openReadmeFile(string):
    markDown = open("Readme.md",'a', newline='')
    markDown.write("\n" + str(string))
    markDown.close()

def maxVolume(numpy):
    return np.max(numpy)

if __name__ =="__main__":
    ts = time.time()
    tS = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    timeStamp = tS
    print("HELLO")
    markDown = open("Readme.md",'a', newline='')
    markDown.write("\n* "+tS+"\n     - | ENVIRONMENT |Episodes / Steps|   ALGORITHM   |   REWARDS   | MAX REWARDS |\n  |---|---| --- | --- | --- |")
    markDown.close()
    testRoutineDQN()
    testRoutineDQNUnityBasic
    testRoutinePPO()
    testRoutineDDPG3DBALL()
    testRoutineDDPGMountainCar