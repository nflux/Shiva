import os
import csv
import pandas as pd
import numpy as np
import datetime
import time

metrics = ["Reward/Per_Step", "Reward/Per_Episode"]
env = ["CartPole-v0"]
algType = ["DQNAlgorithm",]
output = []
def openCSVFile():
    # file = open("Benchmark/"+str(metrics[1])+" " +env[0]+" "+algType[0] +" "+ ""+'.csv', 'w+')
    # return file
    os.chdir("..")
    print (os.path.abspath(os.curdir))
    df=pd.read_csv("Benchmark/"+str(metrics[1])+" " +algType[0]+" "+env[0] +" "+ ""+'.csv', sep=',',header=None)
    print(df.values)
    res = [[ i for i, j in df.values ], [ j for i, j in df.values ]] 
    output.append(np.average(res[1][0:199]))
    output.append(np.average(res[1][0:399]))
    output.append(np.average(res[1][0:599]))
    output.append(np.average(res[1][0:799]))
    output.append(np.average(res[1][0:999]))
    print ("Modified list is : " + str(len(res[1])))
    print ("Modified list2 is : " + str(output))
    os.chdir("benchmarkCode")
    openReadmeFile("       |Cartpole v0|1000 Steps|       **DQN**     |       "+ str(output[0])+": 0-200 Steps, "+ str(output[1])+" : 0-200 Steps, "+ str( output[2])+" : 0-400 Steps, "+ str(output[3])+" : 0-800 Steps, "+ str(output[4])+" : 0-1000 Steps"+"       |")
    # print ("Modified list2 is : " + str(res[1][0:399]))
    # print ("Modified list2 is : " + str(res[1][0:599]))
    # print ("Modified list2 is : " + str(res[1][0:799]))
    # print ("Modified list 2is : " + str(res[1][0:999]))
def openReadmeFile(string):
    markDown = open("Readme.md",'a', newline='')
    markDown.write("\n" + str(string))
    markDown.close()
if __name__ =="__main__":
    ts = time.time()
    tS = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    timeStamp = tS
    print("HELLO")
    markDown = open("Readme.md",'a', newline='')
    markDown.write("\n* "+tS+"\n     - | ENVIRONMENT |Episodes / Steps|   ALGORITHM   |   REWARDS   |\n       |---|---|       ---     |       ---       |")
    markDown.close()
    # openReadmeFile()
    openCSVFile()