from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from neuralmmo_st.forge import trinity
from ...forge.ethyr.torch.param import setParameters, zeroGrads
from ...forge.ethyr.torch import optim
from ...forge.ethyr.rollouts import Rollout

import pygame # for direct human-controlled movement
import os
# import keyboard # needs root access!
from ...forge.blade.systems import ai

from ...forge.blade.action.v2 import ActionV2
import neuralmmo_st.forge.blade.action.v2 as ActionsV2
from ...forge.blade.action.tree import ActionTree

class Sword:
   PROCEDURAL_MOVE = (0,0)
   counter = 0
   isReady = False
   tickOffset = 0
   livingBlue = list([0, 2, 4])
   blueMove = 0
   counterLimit = 6

   def __init__(self, config, args):
      self.config, self.args = config, args
      self.nANN, self.h = config.NPOP, config.HIDDEN
      self.anns  = [trinity.ANN(config)
            for i in range(self.nANN)]

      self.init, self.nRollouts = True, 32
      self.networksUsed = set()
      self.updates, self.rollouts = defaultdict(Rollout), {}
      self.ents, self.rewards, self.grads = {}, [], None
      self.nGrads = 0
      
      ### Pygame init for controls ###
      os.environ["SDL_VIDEODRIVER"] = "dummy"
      pygame.init()
      print('### START pygame list modes ###');
      pygame.display.list_modes()
      print('### END pygame list modes ###');
      pygame.display.set_mode((1920,1080))
      pygame.surface.Surface((1920,1080))

   def backward(self):
      ents = self.rollouts.keys()
      anns = [self.anns[idx] for idx in self.networksUsed]

      reward, val, grads, pg, valLoss, entropy = optim.backward(
            self.rollouts, anns, valWeight=0.25, 
            entWeight=self.config.ENTROPY)
      self.grads = dict((idx, grad) for idx, grad in
            zip(self.networksUsed, grads))

      self.blobs = [r.feather.blob for r in self.rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()

   def sendGradUpdate(self):
      grads = self.grads
      self.grads = None
      return grads
 
   def sendLogUpdate(self):
      blobs = self.blobs
      self.blobs = []
      return blobs

   def sendUpdate(self):
      if self.grads is None:
          return None, None
      return self.sendGradUpdate(), self.sendLogUpdate()

   def recvUpdate(self, update):
      for idx, paramVec in enumerate(update):
         setParameters(self.anns[idx], paramVec)
         zeroGrads(self.anns[idx])

   def collectStep(self, entID, atnArgs, val, reward):
      if self.config.TEST:
          return
      self.updates[entID].step(atnArgs, val, reward)

   def collectRollout(self, entID, ent):
      assert entID not in self.rollouts
      rollout = self.updates[entID]
      rollout.finish()
      self.nGrads += rollout.lifespan
      self.rollouts[entID] = rollout
      del self.updates[entID]

      # assert ent.annID == (hash(entID) % self.nANN)
      self.networksUsed.add(ent.annID)

      #Two options: fixed number of gradients or rollouts
      #if len(self.rollouts) >= self.nRollouts:
      if self.nGrads >= 100*32:
         self.backward()

   def decide(self, ent, stim):
      reward, entID, annID = 0, ent.entID, ent.annID
      action, arguments, atnArgs, val = self.anns[annID](ent, stim)
      
      # TESTING
      print("[sword-AI][entID="+str(entID)+"][ent="+str(ent)+"][annID="+str(annID)+"]arguments:" + str(arguments) + "; action=" + str(action))
      #import forge.blade.action.v2 as ActionsV2
      #action = action[0], ActionsV2.Melee # TEMP
      
      self.collectStep(entID, atnArgs, val, reward)
      self.updates[entID].feather.scrawl(
            stim, ent, val, reward)
      return action, arguments, float(val)

   def decide_CONTROLLED(self, ent, enemy_ent, stim, world, currentTick, allEnts):
      if (not Sword.isReady and Sword.counter > 0 and Sword.counter % Sword.counterLimit == 0):
         inputString = input("Press Enter to keep idling or 0 to start simulation!: ")
         if (inputString == "0"):
            Sword.isReady = True
            Sword.tickOffset = currentTick

      actions = ActionTree(world.env, ent, ActionV2).actions()
      if (Sword.isReady):
         
         ### ANN CODE ###
         ###reward, entID, annID = 0, ent.entID, ent.annID
         ###action_X, arguments, atnArgs, val = self.anns[annID](ent, stim)
         ###print("[sword]ANN-action:" + str(action_X))
         
         ### HUMAN INPUT ###
         # actions = ActionTree(world.env, ent, ActionV2).actions()      
         # #print("[sword]str(actions):" + str(actions))
         # actn = actions[1], ActionsV2.Range # Set attack here
         # print("[sword]Custom-Action: actn=" + str(actn))
         # print("[sword]Custom-Action: pos=" + str(ent.pos))
         
         # if (counter > counterLimit):
            # inputString = input("Movement key: ")
            # PROCEDURAL_MOVE = (0,0)
            # if (inputString == 'j'):
               # PROCEDURAL_MOVE = (0,-1)
            # elif (inputString == 'l'):
               # PROCEDURAL_MOVE = (0,1)
            # elif (inputString == 'i'):
               # PROCEDURAL_MOVE = (-1,0)
            # elif (inputString == 'k'):
               # PROCEDURAL_MOVE = (1,0)
            # print('The inputted string is:' + inputString + '; PROCEDURAL_MOVE=' + str(PROCEDURAL_MOVE))
         # else:
            # PROCEDURAL_MOVE = (0,0)
         ### END HUMAN INPUT ###
         
         ### PROCEDURAL INPUT ###
            
         
         simTick = currentTick - Sword.tickOffset
         # print("[SWORD][entID=" + str(ent.entID) + "]currentTick=" + str(currentTick) + "; simTick = " + str(simTick))
         
         # print("[SWORD]AllEnts=" + str(allEnts))
         if (Sword.blueMove == 0 and int(ent.teamID) == 0):
            currentLivingBlue = list()
            for teammateID in Sword.livingBlue:
               for e in allEnts:
                  if int(e.entID) == teammateID:
                     currentLivingBlue.append(teammateID)
            # print("[SWORD]BlueTeam survivors=" + str(currentLivingBlue))
            if (currentLivingBlue != Sword.livingBlue):
               # print("[SWORD]livingBlue changed. Last value= " + str(Sword.livingBlue))
               Sword.livingBlue = currentLivingBlue
               Sword.blueMove = 2
         
         nearestEnemyAndDistance = ai.findNearestEnemyAndDistance(world, ent, allEnts)
         if (nearestEnemyAndDistance == None): # No more enemies
            arguments = ((0,0), [ent])
            actn = actions[1], ActionsV2.Range
         else:            
            nearestDist = nearestEnemyAndDistance[0]
            nearestEnemy = nearestEnemyAndDistance[1]
            # print("[SWORD][entID=" + str(ent.entID) + "] Nearest Enemy and Distance = " + str((nearestEnemyAndDistance[0], nearestEnemyAndDistance[1].entID)))
            
            attackAction = ActionsV2.Melee
            if (nearestDist <= self.config.MELEERANGE):
               # Do melee attack
               attackAction = ActionsV2.Melee
               enemy_ent = nearestEnemy
               # print("[SWORD][entID=" + str(ent.entID) + "] MELEE: " + str(nearestDist) + "<=" + str(self.config.MELEERANGE) + "; targ=" + str(enemy_ent.entID))
            elif (nearestDist <= self.config.RANGERANGE):
               # Do range attack
               attackAction = ActionsV2.Range
               enemy_ent = nearestEnemy
               # print("[SWORD][entID=" + str(ent.entID) + "] RANGE: " + str(nearestDist) + "<=" + str(self.config.RANGERANGE) + "; targ=" + str(enemy_ent.entID))
            else:
               # No attack
               enemy_ent = ent
               # print("[SWORD][entID=" + str(ent.entID) + "] NOATTACK: " + str(nearestDist) + "; targ=" + str(enemy_ent.entID))
            
            actn = actions[1], attackAction # Set attack
            
            movements = self.GetMovementList(int(ent.entID))
            if (simTick >= len(movements)):
               if (int(ent.teamID) == 0 and Sword.blueMove > 0):
                  # print("[SWORD] Move " + str(ent.entID) + " to the right")
                  Sword.PROCEDURAL_MOVE = (0,1)
                  if (int(ent.entID) == Sword.livingBlue[-1]):
                     Sword.blueMove -= 1
                     # print("[SWORD]decrement blueMove to: " + str(Sword.blueMove))
               else:
                  Sword.PROCEDURAL_MOVE = (0,0)
            else:
               Sword.PROCEDURAL_MOVE = movements[simTick]
            ### END PROCEDURAL INPUT ###
            
            arguments = (Sword.PROCEDURAL_MOVE, [enemy_ent]) # set movement amount here and enemy here. If enemy_ent is not self, then no attack takes place.
            # print("[sword-HUMAN]arguments:" + str(arguments))
      else:
         arguments = ((0,0), [ent])
         actn = actions[1], ActionsV2.Range

      Sword.counter += 1

      return actn, arguments

   def GetMovementList(self, id):
      switcher = {
         0: [(0,0),(-1,0),(-1,0),(0,1), (0,1), (0,0), (0,0),(0,0)],
         1: [(0,0),(0,0),(0,0),(0,1), (1,0), (1,0), (1,0), (0,0)],
         2: [(0,0),(0,1), (-1,0),(-1,0),(0,-1),(0,-1),(0,0),(0,0)],
         3: [(0,0),(0,0),(0,0),(0,1), (0,1),(1,0), (1,0),(0,0)],
         4: [(0,0),(0,0), (0,-1),(-1,0),(-1,0),(0,0), (0,0),(0,0)],
         5: [(0,0),(0,0),(0,0),(0,1), (0,1), (1,0), (1,0), (1,0)]
      }
      return switcher.get(id, [])
   
   