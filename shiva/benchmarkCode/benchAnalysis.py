

def openCSVFile():
    file = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'w+')
    return file
def openReadmeFile():
    markDown = open("benchmarkCode/Readme.md",'a', newline='')
    return markDown
if __name__ =="__main__":
    print("HELLO")