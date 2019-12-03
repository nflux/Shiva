#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from HFO.hfo.hfo import *
from HFO.bin import Communicator

# class Trainer:
#     def __init__(self):
#         self._coachPort = 2001

#     def initComm(self):
#         """ Initialize communication to server. """
#         self._comm = Communicator.ClientCommunicator(port=self._coachPort)
#         self.send('(init (version 8.0))')
#         self.checkMsg('(init ok)', retryCount=5)
#         # self.send('(eye on)')
#         self.send('(ear on)')

#     def send(self, msg):
#         """ Send a message. """
#         self._comm.sendMsg(msg)
    
#     def recv(self, retryCount=None):
#         """ Recieve a message. Retry a specified number of times. """
#         return self._comm.recvMsg(retryCount=retryCount).strip()

#     def checkMsg(self, expectedMsg, retryCount=None):
#         """ Check that the next message is same as expected message. """
#         msg = self.recv(retryCount)
#         if msg != expectedMsg:
#             sys.stderr.write('Error with message')
#             sys.stderr.write('  expected: ' + expectedMsg)
#             sys.stderr.write('  received: ' + msg)
#             # print >>sys.stderr,len(expectedMsg),len(msg)
#             raise ValueError

    

def main(trainer):
  # Create the HFO Environment
#   hfo = HFOEnvironment()
#   trainer = Trainer()
#   trainer.initComm()

  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
#   hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
#                       'HFO/bin/teams/base/config/formations-dt', 6000,
#                       'localhost', 'base_left', False)

  for episode in itertools.count():
    status = IN_GAME
    while status == IN_GAME:
      # Grab the state features from the environment
    #   features = hfo.getState()
      # Get any incoming communication
    #   msg = hfo.hear()
    #   # Print the incoming communication
    #   if msg:
    #     print(('Heard: %s'% msg))
      # Take an action
      trainer.send('(move (ball) -10 10 10 10 10)')
    #   hfo.act(DASH, 20.0, 0.)
      # Create outgoing communication
      # the message can contain charachters a-z A-Z 0-9 
      # the message can contain special charachters like ?SPACE-*()+_<>/
      # the message cannot contain !@#$^&={}[];:"'
    #   hfo.say('Hello World')

      # Advance the environment and get the game status
    #   status = hfo.step()
    # Check the outcome of the episode
    # print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down
    # if status == SERVER_DOWN:
    #   hfo.act(QUIT)
    #   exit()

if __name__ == '__main__':
  main()