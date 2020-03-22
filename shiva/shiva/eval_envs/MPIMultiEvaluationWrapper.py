import sys, time,traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import numpy as np
import pandas as pd
import math
import random

from shiva.core.admin import logger
from shiva.eval_envs.Evaluation import Evaluation
from shiva.helpers.misc import terminate_process
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin
from shiva.envs.Environment import Environment

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("shiva")

class MPIMultiEvaluationWrapper(Evaluation):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.sort = False #Flag for when to sort
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEvaluationWrapper, self).__init__(self.configs)
        self._launch_evals()
        self.num_envs = self.configs['Environment']['num_instances']
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self.rankings = np.zeros(self.num_agents)
        self.evaluations = dict()
        self.info = MPI.Status()
        self.log("This at 35 is not printed")
        self.agent_ids = self.meta.bcast(None,root=0)
        self.agents = [Admin._load_agents(self.eval_path+'Agent_'+str(agent_id))[0] for agent_id in self.agent_ids]
        # self.io = MPI.COMM_WORLD.Connect(self.evals_io_port, MPI.INFO_NULL)
        # self._io_load_agents()
        print("AGENT HELLO WORLD", self.agents[0].eloReward)
        self.initial_agent_selection()

        # THE START OF INITIATE MATCHING FUNCTION PIPELINE TEST
        # self.log('Initiate Matching Function Test')
        # self.eloProbability(100,200,10)
        # self.CalculateEloReward(100,.50,10,1)
        # # self.Matcher(self.agent_ids, 5, [0,1,2,3,4], 0.5, 100,.10)
        # reward = np.array([10,10,10,10,10])
        # lists = self.Matcher(self.agent_ids, 5, reward,.5, 5,.10)
        # print("SELF MATCHER", lists[1])
        # self.log('End of Matching Function Test')

        # THE END OF MATCHING FUNCTION PIPELINE TEST
        self.run()

    def run(self):
        self.log('MultiEvalWrapper is running')
        self._get_initial_evaluations()

        while True:
            time.sleep(0.001)
            self._get_evaluations(True)

            if self.sort:
                self.rankings = np.array(sorted(self.evaluations, key=self.evaluations.__getitem__,reverse=True))
                print('Rankings: ', self.rankings)
                print('Rankings type: ', type(self.rankings))
                self.meta.send(self.rankings,dest= 0,tag=Tags.rankings)
                print('Sent rankings to Meta')
                self.sort = False

        self.close()

    def _launch_evals(self):
        # Spawn Single Environments
        self.evals = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvaluation.py'], maxprocs=self.num_evals)
        self.evals.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        #self.log('Eval configs sent')
        eval_spec = self.evals.gather(None, root=MPI.ROOT)  # Wait for Eval Specs ()
        self.log("These are the evals {}".format(eval_spec))
        #self.log('Eval specs received')
        assert len(eval_spec) == self.num_evals, "Not all Evaluations checked in.."
        self.eval_specs = eval_spec[0] # set self attr only 1 of them




    def _get_meval_specs(self):
        return {
            'type': 'MultiEval',
            'id': self.id,
            'eval_specs': self.eval_specs,
            'num_evals': self.num_evals
        }

    def _get_evaluations(self,sort):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evals):
            agent_id = self.evals.recv(None, source=MPI.ANY_SOURCE, tag=Tags.evals, status=self.info)
            env_source = self.info.Get_source()
            evals = self.evals.recv(None, source=env_source, tag=Tags.evals)
            self.evaluations[agent_id] = evals.mean()
            self.sort = sort
            print('Multi Evaluation has received evaluations!')
            self.agent_selection(env_source)

    #def _get_evaluations(self,sort):
        #if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag = Tags.evals):
            #evals = self.evals.recv(None,source=MPI.ANY_SOURCE,tag=Tags.evals,status=self.info)
            #eval_source = self.info.Get_source()
            #self.log('Agent IDS: '.format(evals['agent_ids']))
            #for i in range(len(evals['agent_ids'])):
                #self.evaluations[evals['agent_ids'][i]] = evals['evals'][i].mean()
            #self.sort=sort
            #print('Multi Evaluation has received evaluations!')
            #self.agent_selection(eval_source)

    def _get_initial_evaluations(self):
        while len(self.evaluations ) < self.num_agents:
            self._get_evaluations(False)
            #print('Multi Evaluations: ',self.evaluations)

    def initial_agent_selection(self):

        # System Using Team Matching 
        reward = np.full(self.configs['Environment']['num_instances'],10)
        for x in range(len(self.agent_ids)):
            reward[x] = self.agents[self.agent_ids[x]].eloReward
        ag = self.Matcher(self.agent_ids, self.configs['Environment']['num_instances'], reward,self.configs['Matchmaker']['probability'], self.configs['Matchmaker']['constant'],self.configs['Matchmaker']['scaler'])
        
        # dictionaryAgents = self.Matcher(self.agent_ids, 5, reward,.5, 5,.10)
        for x in range(len(ag)):
            print("X Agents:", ag[x])
            # if( len(ag) == 1):
            #     self.agent_sel = [ag[x]]
            # else :    
            #     self.agent_sel = ag[x]
            self.agent_sel = ag[x]
            print('Selected Evaluation Agents for Environment {}: {}'.format(x, self.agent_sel))
            self.evals.send(self.agent_sel,dest=x,tag=Tags.new_agents)

        # Randomly selecting Agents
        # for i in range(self.num_evals):
        #     self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))[0]
        #     print('Selected Evaluation Agents for Environment {}: {}'.format(i, self.agent_sel))
        #     self.evals.send(self.agent_sel,dest=i,tag=Tags.new_agents)

    #Agent Selection currently randomly Selects an agent. 
    def agent_selection(self,env_rank):

        reward = np.full(self.configs['Environment']['num_instances'],10)
        for x in range(len(self.agent_ids)):
            reward[x] = self.agents[self.agent_ids[x]].eloReward
        ag = self.Matcher(self.agent_ids, self.configs['Environment']['num_instances'], reward,self.configs['Matchmaker']['probability'], self.configs['Matchmaker']['constant'],self.configs['Matchmaker']['scaler'])
        if(len(ag[0]) == 1):
            self.agent_sel = [ag[random.randrange(len(ag))]]
        else:
            self.agent_sel = ag[random.randrange(len(ag))]
        # self.agent_sel = [ag[random.randrange(len(ag))]]
        #Randomly Selects and Agent
        #self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))
        print('Selected Evaluation Agents for Environment {}: {}'.format(env_rank, self.agent_sel))
        self.evals.send(self.agent_sel,dest=env_rank,tag=Tags.new_agents)


        
    def eloProbability (self, rating1, rating2):
        self.probability = 1.0 * 1.0 / (1 + 1.0 * math.pow (10, 1.0 * (rating1 - rating2)))
        self.log("ELO Probability Score : {}", self.probability)
        print("ELO Probability Score : ", self.probability)
        return self.probability
    

    # rating1 - Score for average Agent/Team1
    # rating2 - Score for Agent/Team2
    # n - Determine How large the score it would take to effect the overall score
    # Probability of rating2 



    # reward - A current Agent/Team's average Reward 
    # Elo Probability - A Current Agent/Team's Probability Reward
    # K - some kind of constant
    # w - 1 when A Team/Agent had won
    # w - 0.5 when A Team/Agent had Tied
    # w - 0 when A Team/ Agent had Lost

    def CalculateEloReward(self, reward, eloProbability, k, w):
        self.reward = reward + k *(w - eloProbability)
        self.log('Elo Reward : {}', self.reward )
        print('Elo Reward : ', self.reward )
        return self.reward

    # team1 - Average Reward for Team1 or Agent1
    # team2 - Average Reward for Team2 or Agent2
    # n - How high a constant n, should be to make a significant change to a team's score
    # The Probability of Team2 winning.

    def TeamEloRating(self, team1, team2):
        self.rating = self.eloProbability(team1, team2)
        self.log("Probability Calculation Done. Probability of Team2 wiining : {}", self.rating)
        return self.rating

    # TeamCompare
    # Requires minumum of Two Teams. 
    # Assumes team1Rewards is summed.
    # Assumes team2Rewards is summed.
    # r: probability Threshold
    # scoreFactor: The Constant Score.

    def TeamCompareProbability(self,team1Rewards,team2Rewards, r, scoreFactor):
        # boolResult = False
        prob = max(self.eloProbability(team1Rewards, team2Rewards, scoreFactor), self.eloProbability(team2Rewards, team1Rewards, scoreFactor))
        print("TeamCompareProbability r: ",r)
        print("TeamCompareProbability prob: ", prob)
        if (prob <= r):
            boolResult = True
        else:
            boolResult = False
        return boolResult

    


# Using More of an ELO based matcher WITH RANDOM Generator instead of Ordinal.
# Need to have Incremental differential range of ELO Ratings
# EX: if agent ELO score or probability is 100 incremental 10, would mean try to match with 90 or 110, but if not found increase another 10. 
# EX: Compare just the ELO Probability instead. 
# Rewards contains the list of Rewards that Agent contains.
# n = The number of Teams to create and divide among.
# r = Probability Threshold of the winning versus lossing from the max of the two differences. 
# scoreFactor = constant Score that you can get. 
# increaseScale = the Rate of an increasing scale increases r at a rate.
# 

    def Matcher(self,agents, n, rewards, r, scoreFactor,increaseScale):
        if((len(agents) % n) != 0):
            print("WARNING Not enough Agents to make the Request Teams. Skipping EloMatcher")
            print("CONSIDER using random matcher. ")
            pass
        
        random.seed()
        teamCreated = False
        # rand = random.randrange(len(agents) - 1)
        self.agentID =[]
        self.orignalR = r
        self.assignedOrNot = np.full(agents.shape,False)
        self.agentIDDict= {}
        for x in agents :
            self.agentIDDict.update({x : False})
       
        #TODO Replace assignedOrNot with a Dictionary. 
        print(self.assignedOrNot)
        self.teams = []
        self.totalRewards = []
        self.teamLength = len(agents)/n
        print("n :", n)
        print("Length of Agents ", len(agents) )
        breaker = False
        r = r
        # p: Agent A To Match
        # p = 0
        # k: The Rest of the Agents to try to to match to 
        # t: the Number of Teams that has been made. 
        # ag: Agent counter
        p = random.randrange(len(agents))
        k = 0
        t = 0
        ag = 1
        self.status = []
        while (breaker == False):
            print("SUM",np.sum(self.assignedOrNot))
            
            # Go to the Next P-agent when it is assigned or not.
            if(self.assignedOrNot[p]):
                while(self.assignedOrNot[p] and False in self.assignedOrNot):
                    random.seed()
                    p = random.randrange(len(agents))
                    print("STATUS In Assigned Or Not", self.assignedOrNot)
                    print("STATUS In Assigned Or Not on p", p)
                    print("Assigned Or Not" , self.assignedOrNot[p])
                    
            if( ( ((np.sum(self.assignedOrNot) % self.teamLength) == 0 )) and (t != n) and teamCreated == False):
                # Append a New Team 
                for x in range(t,t+1):
                    self.status.append([])
                    self.teams.append([])
                    self.agentID.append([])
                    self.totalRewards.append([])
                    for y in range (0, int(self.teamLength)):
                        self.teams[x].append(agents[p])
                        self.agentID[x].append(p)
                        self.totalRewards[x].append(rewards[p])
                    

                # Increment T-index to track how many teams have been have
                # NOTE: May have one-off error. 
                print("TEAM", t)
                teamCreated = True
                t = t + 1
                ag = 1
            
            if(k< len(agents) and self.assignedOrNot[p] == False):
                print("K",k)
                probMatching = max(self.eloProbability(rewards[k], rewards[p], scoreFactor), self.eloProbability(rewards[p], rewards[k], scoreFactor))
            print("Counter K: ",k)
            print("Counter P: ",p)
            print("Counter T:" ,t)
            print("Counter n:", n)
            print("Counter r:", r)
            print("Counter ag:", ag)
            print("Team Length", self.teamLength)
            print(self.assignedOrNot)
            if( self.teamLength > 1 and k != p and k< len(agents) and probMatching <= r and (self.assignedOrNot[k] == False)  and (self.assignedOrNot[p] == False)):
                print("Hello WORLD1")
                # print(t)
                # print(self.totalRewards)
                self.teams[t-1][ag] = (agents[k])
                self.totalRewards[t-1][ag] = (rewards[k])
                self.agentID[t-1][ag] = k
                self.agentIDDict[agents[p]] = True
                self.agentIDDict[agents[k]] = True
                self.assignedOrNot[p] = True
                self.assignedOrNot[k] = True
                p= random.randrange(len(agents))
                teamCreated = False
                k = 0
                ag = ag + 1


            elif( 1 == self.teamLength and k< len(agents)  and  (self.assignedOrNot[p] == False)):
                print("Hello WORLD2")
                # print(t)
                # print(self.totalRewards)
                self.assignedOrNot[p] = True
                self.agentIDDict[agents[p]] = True
                # self.teams[t-1][ag] = (agents[p])
                p= random.randrange(len(agents))
                teamCreated = False
                k = 0
                ag = ag + 1

            # If the Number of teams is an Odd Number
            elif(  k != p and k< len(agents)  and  (self.assignedOrNot[p] == False) and np.sum(self.assignedOrNot) < len(agents) and (self.teamLength % 2 != 0)):
                print("Hello WORLD3")
                # print(t)
                # print(self.totalRewards)
                self.assignedOrNot[p] = True
                self.agentIDDict[agents[p]] = True
                self.teams[t-1][ag] = (agents[p])
                p= random.randrange(len(agents))
                teamCreated = False
                k = 0
                ag = ag + 1
            
            # If It ran through the entire list of agents and no match is found expand probability of success number by factor of orignalR
            elif ((self.assignedOrNot[p] == False) and k == len(agents)):
                print("Increase R")
                r = r + self.orignalR *increaseScale
                print("Increase R", r)
                k = 0
        # When the Number of Teams needed is met.
            elif( (False in self.assignedOrNot) == False):
    #             breaker = False
                e = 0
                print("breaker Main", breaker)
                while (e < t):
                    if( e+1 != t):
                        breaker = self.TeamCompareProbability(sum(self.totalRewards[e]),sum(self.totalRewards[e+1]),r, scoreFactor)
                        print("breaker", breaker)
                        print("e", e)
                        print("FINAL Assigned OR Not", self.assignedOrNot)
                        print("FINAL DICT Assigned Or Not",self.agentIDDict )
                    if(breaker == False and n != 1):
                        #Needs to Scramble and Restart Search. 
                        # breaker = False
                        localE = e
                        while(e < (localE + 1) and n != 1):
                            i = 0
                            while(i< len(self.teams[e])):
                                self.assignedOrNot[self.teams[e][i]] = False
                                i = i + 1
                            print("POP Confirmed")
                            # self.teams.pop(e)
                            # self.teams.remove(e)
                            del self.teams[e]
                            e = e + 1
                            ag = 1
                            t = t - 1
                        #OBTAIN UI
    #                   return breaker  
    #  To be changed with Rescramble once reimplemented into pipeline. 
                        print("FINAL Assigned OR Not", self.assignedOrNot)
                        print("FINAL DICT Assigned Or Not",self.agentIDDict )
                        e = t
                        
                    if(n == 1):
                        breaker = True

                    else:
                        print("FINAL Assigned OR Not", self.assignedOrNot)
                        print("FINAL DICT Assigned Or Not",self.agentIDDict )
                        print(self.teams)
    #                     return breaker
                    e = e + 1
            else:
            # Even if no agent is assigned move on to the next agent to check. 
                k = k + 1
    
            
                

        
        # IF MATCHING FAILS REMATCH. 
        print("FINAL Assigned OR Not", self.assignedOrNot)
        print("FINAL DICT Assigned Or Not",self.agentIDDict )
        print("THE FINAL TEAMS", self.teams)
        return self.teams

    def eloRewarder(self, teamAlpha, teamAlphaRewards, teamBravo, teamBravoRewards, k, wAlpha, wBravo):
        self.teamAlphaProbabilityWin = self.TeamEloRating(teamBravoRewards, teamAlphaRewards)
        self.teamBravoProbabilityWin = self.TeamEloRating(teamAlphaRewards, teamBravoRewards)
        # teamNewAlphaRewards = calculateEloReward(teamAlphaRewards,teamAlphaProbabilityWin, k, wAlpha)
        # teamNewBravoRewards = CalculateEloReward(teamBravoRewards, teamBravoProbabilityWin, k ,wBravo)


        for x in range(len(teamAlpha)):s
            self.teamAlphaRewards[x] = self.calculateEloReward(teamAlphaRewards, self.teamAlphaProbabilityWin, k, wAlpha)
            #self.agents[teamAlpha[x]].eloReward = calculateEloReward(teamAlphaRewards, teamAlphaProbabilityWin, k, wAlpha)

        for y in range(len(teamBravo)):
            self.teamBravoRewards[y] = self.calculateEloReward(teamAlphaRewards, self.teamAlphaProbabilityWin,k, wBravo)
            #self.agents[teamAlpha[y]].eloReward = calculateEloReward(teamAlphaRewards, teamAlphaProbabilityWin, k, wAlpha)

        return self.teamAlphaRewards, self.teamBravoRewards
        #return  teamAlpha, teamBravo

            

            
        
    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'MultiEval {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))



if __name__ == "__main__":
    try:
        MPIMultiEvaluationWrapper()
    except Exception as e:
        print("Eval Wrapper error:", traceback.format_exc())
    finally:
        terminate_process()
