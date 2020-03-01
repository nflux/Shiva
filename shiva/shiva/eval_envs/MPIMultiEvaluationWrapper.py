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
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self.rankings = np.zeros(self.num_agents)
        self.evaluations = dict()
        self.info = MPI.Status()
        self.log("This at 35 is not printed")
        self.agent_ids = self.meta.bcast(None,root=0)
        print('Agent IDS: ', self.agent_ids)
        self.initial_agent_selection()
        self.log('Initiate Matching Function Test')
        self.eloProbability(100,200,10)
        self.CalculateEloReward(100,.50,10,1)
        self.Matcher(self.agents_ids, 100, 10, 10, 10)
        self.log('End of Matching Function Test')
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
        for i in range(self.num_evals):
            self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))[0]
            print('Selected Evaluation Agents for Environment {}: {}'.format(i, self.agent_sel))
            self.evals.send(self.agent_sel,dest=i,tag=Tags.new_agents)


    #Agent Selection currently randomly Selects an agent. 
    def agent_selection(self,env_rank):
        self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))
        print('Selected Evaluation Agents for Environment {}: {}'.format(env_rank, self.agent_sel))
        self.evals.send(self.agent_sel,dest=env_rank,tag=Tags.new_agents)

    # rating1 - Score for average Agent/Team1
    # rating2 - Score for Agent/Team2
    # n - Determine How large the score it would take to effect the overall score
    # Probability of rating2 
    def eloProbability (self, rating1, rating2, n):
        self.probability = 1.0 * 1.0 / (1 + 1.0 * math.pow (10, 1.0 * (rating1 - rating2)))
        self.log("ELO Probability Score : {}", self.probability)
        return self.probability
    

    # reward - A current Agent/Team's average Reward 
    # Elo Probability - A Current Agent/Team's Probability Reward
    # K - some kind of constant
    # w - 1 when A Team/Agent had won
    # w - 0.5 when A Team/Agent had Tied
    # w - 0 when A Team/ Agent had Lost
    def CalculateEloReward(self, reward, eloProbability, k, w):
        self.reward = reward + k *(w - eloProbability)
        self.log('Elo Reward : {}', self.reward )
        return self.reward

    # team1 - Average Reward for Team1 or Agent1
    # team2 - Average Reward for Team2 or Agent2
    # n - How high a constant n, should be to make a significant change to a team's score
    # The Probability of Team2 winning.
    def TeamEloRating(self, team1, team2, n):
        self.rating = self.eloProbability(team1, team2, n)
        self.log("Probability Calculation Done. Probability of Team2 wiining : {}", self.rating)
        return self.rating

# Using More of an ELO based matcher WITH RANDOM Generator instead of Ordinal.
# Need to have Incremental differential range of ELO Ratings
# EX: if agent ELO score is 100 incremental 10, would mean try to match with 90 or 110, but if not found increase another 10. 
# EX: Compare just the ELO Probability instead. 
# Rewards contains the list of Rewards that Agent contains.
# r = Probability Threshold of the winning versus lossing from the max of the two differences. 


    def Matcher(self, agents, n, rewards, r, scoreFactor, increaseRate):
        agent_id = self.evals.recv(None, source=MPI.ANY_SOURCE, tag=Tags.evals, status=self.info)
        if((len(agents) % n != 0)):
            self.log("WARNING Not enough Agents to make the Request Teams. Skipping EloMatcher")
            self.log ("CONSIDER using random matcher. ")
            pass
        self.random.seed()
        self.teamCreated = False
        self.rand = self.random.randrange(len(agents) - 1)
        self.agentID = []
        self.originalR = r
        self.assignedOrNot = np.full(agents.shape, False)
        self.log('Currently what is assigned or not', assignedOrNot)
        self.teams = []
        self.totalRewards = []
        self.teamLength = len(agents)%n
        # Breaks become false when Teams are made (INVERSE RESULT!!!!)
        breaker = True
        '''
        p: Agent A to Match.
        k: The rest of the Agents to try to match to.
        t: The number of Teams that has been made. 
        ag: Agent counter, to start when a new P has been assigned. 
        '''
        p = random.randrange(len(agents)-1)
        k = 0
        t = 0
        ag = 1
        while (breaker):
            self.log("SUM", np.sum(assignedOrNot))
              # Go to the Next P-Agent when it is assigned or not.
            if (assignedOrNot[p]):
                while(assignedOrNot[p] and False in assignedOrNot):
                    p = random.randrange(len(agents) - 1)
                    self.log("Assigned Or Not" , assignedOrNot[p])
            if(((np.sum(assignedOrNot) % n) == 0) and (t != n) and teamCreated == False ):
                # Appends a New Team
                for x in range(t,t+1):
                self.status.append([])
                self.teams.append([])
                self.agentID.append([])
                self.totalRewards.append([])
                for y in range (0, int(teamLength)):
                    self.status[x].append(1)
                    self.teams[x].append(agent[p])
                    self.agentID[x].append(p)
                    self.totalRewards[x].append(rewards[p])
                self.log('Team #: ', t)
                teamCreated = True
                t = t + 1
                ag = 1
            if(k < len(agents) and assignedOrNot[p] == False):
                probMatching = max(Probability(rewards[k], rewards[p], scoreFactor), Probability(rewards[p], rewards[k], scoreFactor))
                self.log("Check Assignment status on Counter K:", k)
            self.log("Counter k:", k)
            self.log("Counter p:", p)
            self.log("Counter t:", t)
            self.log("Counter n:", n)
            self.log("Counter r:", r)
            self.log("Check assignment Status", assignedOrNot)
            if(k != p and k< len(agents) and probMatching <= r and (assignedOrNot[k] == False)  and (assignedOrNot[p] == False)):
                self.log("Reached To Assign a new Agent to a Team.")
                self.log("Team Counter t:", t)
                #self.log("Total ", totalRewards)
                self.teams[t-1][ag] = (agent[k])
                self.totalRewards[t-1][ag] = (rewards[k])
                self.log("Total Rewards at the time", totalRewards)
                self.agentID[t-1][ag] = k
                self.assignedOrNot[p] = True
                self.assignedOrNot[k] = True
                p = random.randrange(len(agents) - 1)
                self.teamCreated = False
                k = 0
                ag = ag + 1
                # If It ran through the entire list of agents and no match is found expand probability of success number by factor of orignalR
            elif ((assignedOrNot[p] == False) and k == len(agents)):
                self.log("Increase R")
                self.log("Original R", originalR)
                self.log("Increase Rate", increaseRate)
                self.log("Old R", r)
                r = r + orignalR * increaseRate
                self.log("New R", r)
                k = 0
            elif( (False in assignedOrNot) == False):
                e = 0
                self.log("Breaker Main", breaker)
            while (e < t):
                if( e+1 != t):
                    breaker = self.TeamCompareProbability(sum(totalRewards[e]),sum(totalRewards[e+1]),r, scoreFactor)
                    self.log("breaker", breaker)
                    self.log("e", e)
                if(breaker == False):
                    #Needs to Scramble and Restart Search. 
                    breaker = False
                    e = t
                    #OBTAIN UI
                    #  return breaker  
                    #  To be changed with Rescramble once reimplemented into pipeline. 
                    return self.teams
                    #   e = e + 1
                else:
                    print(self.teams)
                    #   return breaker
                e = e + 1
        else:
            # Even if no agent is assigned move on to the next agent to check. 
            k = k + 1
`
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
