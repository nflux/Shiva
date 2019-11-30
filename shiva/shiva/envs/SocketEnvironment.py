import gym
from .Environment import Environment
import numpy as np
import torch
import socket
import time 


'''
        NOTES

    THIS ENVIRONMENT WAS ORGINALLY USED AS A HACK TO INTERCEPT OBSERVATIONS AND 
    INJECT ACTIONS INTO UNITY WITH THE MLAGENTS.

    THIS CAN BE MODIFED TO RECEIVE OBSERVATIONS AND ACTIONS FROM A NEW ENVIRONMENT
    THAT DOESN'T HAPPEN TO HAVE AN API.


'''


class SocketEnvironment(Environment):
    def __init__(self,environment):
        super(SocketEnvironment,self).__init__(environment)

        self.env = self.env_name
        # self.obs = self.reset()
        self.rews = 0
        self.world_status = False
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.action_space_continuous = None
        self.action_space_discrete = None 
        self.step_count = 0

        # create a socket object 
        self.s = socket.socket()          
        print("Socket successfully created")
        # Bind to the port
        self.s.bind(('', self.port))
        print("socket binded to {}".format(self.port))
        # put the socket into listening mode 
        self.s.listen(10)      
        print("socket is listening")

    def step(self,action):  
         
        # split the action
        action1 = bytes(str(action) + ' ', "utf-8")
        # action2 = bytes(str() + " ", "utf-8")

        # send back the actions
        self.clientSocket.send(action1)
        # self.clientSocket.send(action2)

        # time.sleep(0.5)

        srd = self.clientSocket.recv(1024)

        # print("Unity Environment srd:", srd)

        # parse srd so we get the reward, done, and next state 
        srd = str(srd).strip('\'').split(' ')
        # print(srd)
        # print(srd[9])
        # this will require more string manipulation

        # not doing much with this currently
        # agent_id = srd[:1]

        self.next_observation = self.bytes2numpy(srd[1],srd[2], srd[3:6], srd[6:9])
        self.rews = np.float32(srd[9])
        self.world_status = np.bool(srd[10])
        self.aver_rew = np.float32(srd[11])

        self.step_count +=1

        if self.normalize:
            return self.next_observation, self.normalize_reward(), self.world_status, {'raw_reward': self.rews}
        else:
            return self.next_observation, self.rews, self.world_status, {'raw_reward': self.rews}

    def reset(self):
        pass

    def set_observation_space(self):

        self.observation_space = 4
        return 8

    def set_action_space(self):
        self.action_space = 2
        return 2

    def get_observation(self):
        # Establish connection with client. 
        self.clientSocket, self.address = self.s.accept()
        s = str(self.clientSocket.recv(1024)).strip('\'').split(' ')
        return self.bytes2numpy(s[1],s[2], s[3:6], s[6:9])

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        # Close the connection with the client
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        try:
            delattr(self, 'clientSocket')
            delattr(self, 'address')
            delattr(self, 's')
        except AttributeError:
            pass

    def bytes2numpy(self, a1, a2, p, v):
        # Turn everything into np floats and strip off string characters
        z = np.float32(a1)
        x = np.float32(a2)
        pos = np.array([np.float32( s.strip('(').strip(')').strip(',') ) for s in p ], dtype=np.float32)
        vel = np.array([np.float32( s.strip('(').strip(')').strip(',') ) for s in v ], dtype=np.float32)
        return np.array([z, x, *list(pos), *list(vel)])







'''

This is the modified C# script used where a client was added that allowed communication
between the MLAgents and Shiva


using UnityEngine;
using MLAgents;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
    public int id;
    private Rigidbody m_BallRb;
    private ResetParameters m_ResetParams;
    private float z1;
    private float x1;       // these are the private variables
    private Vector3 pos1;
    private Vector3 vel1;


    private int step_count;
    private float totalReward;
    private float average_reward_per_episode;

    public override void InitializeAgent()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        var academy = FindObjectOfType<Academy>();
        m_ResetParams = academy.resetParameters;
        SetResetParameters();
    }

    public override void CollectObservations()
    {
        // print("Hello World");
        z1 = gameObject.transform.rotation.z;
        x1 = gameObject.transform.rotation.x;           /// Here I grab them
        pos1 = (ball.transform.position - gameObject.transform.position);
        vel1 = m_BallRb.velocity;
        AddVectorObs(gameObject.transform.rotation.z);
        AddVectorObs(gameObject.transform.rotation.x);
        AddVectorObs(ball.transform.position - gameObject.transform.position);
        AddVectorObs(m_BallRb.velocity);

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f); // mlagents brain action

        //print("BrainZ: " + actionZ);
        //print("BrainX: " + actionZ);

        string state = z1.ToString() + " " + x1.ToString() + " " +  pos1.ToString() + " " + vel1.ToString();   // here i grab the state from private variables
        string actions = "";
        string next_state = "";
        float reward = 0.0f;
        bool done = false;

        // socket goes here
        // Data buffer for incoming data.  
        byte[] bytes = new byte[1024];

        // Connect to a remote device.  
        try {
            // Establish the remote endpoint for the socket.  
            // This example uses port 11000 on the local computer.  
            IPHostEntry ipHostInfo = Dns.GetHostEntry (Dns.GetHostName ());
            IPAddress ipAddress = ipHostInfo.AddressList[0];
            IPEndPoint remoteEP = new IPEndPoint (ipAddress, 12345);

            // Create a TCP/IP  socket.  
            Socket sender = new Socket (ipAddress.AddressFamily,
                SocketType.Stream, ProtocolType.Tcp);
            // print(id);
            // Connect the socket to the remote endpoint. Catch any errors.  
            try {
                sender.Connect (remoteEP);

                // print ("Socket connected to {0}" + sender.RemoteEndPoint.ToString ());

                // Encode the data string into a byte array.  
                byte[] obs1 = Encoding.ASCII.GetBytes (id.ToString() + " " + state);

                // print("Made the obs");
                // Send the data through the socket.  
                sender.Send(obs1);       // here's the first obs i send     
                // print("Sent the obs");

                // Receive the response from the remote device.  
                int bytesRec = sender.Receive(bytes);
                actions = Encoding.ASCII.GetString(bytes, 0, bytesRec).Replace("[", "").Replace("]", "");
                float[] actions_arr = StringToArray(actions, " ");

                actionZ = actions_arr[0]; // Here we overwrite the Tensorflow Brain actions
                actionX = actions_arr[1]; 
                // print(actionZ);
                // print(actionX);

                // so it may not always change, it probably learns not to let the velocity get negative, downwards? like falling velocities
                // it may learn to try to avoid those and the paths that lead up to that 
                
                if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
                    (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
                {
                    gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
                }

                if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
                    (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
                {
                    gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
                }

                float z2 = gameObject.transform.rotation.z;
                float x2 = gameObject.transform.rotation.x;
                Vector3 pos2 = (ball.transform.position - gameObject.transform.position);
                Vector3 vel2 = m_BallRb.velocity;

                next_state = z2.ToString() + " " + x2.ToString() + " " +  pos2.ToString() + " " + vel2.ToString();

                if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
                    Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
                    Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f) {

                    done = true;
                    reward = -1f;
                    totalReward += reward;
                    print(totalReward);
                    average_reward_per_episode = totalReward / step_count;
                    // print(step_count);
                    Done();
                    SetReward(-1f);
                    AgentReset();

                } else {
                    done = false;
                    reward = 0.1f;
                    step_count++;
                    // print(step_count);
                    totalReward += reward;
                    SetReward(0.1f);
                }

                byte[] srd = Encoding.ASCII.GetBytes(
                    id.ToString() + " " +         // the id doesn't seem to matter at the moment
                    next_state + " " +
                    reward.ToString() + " " +
                    done.ToString() + " " +
                    average_reward_per_episode.ToString()
                );

                // send back done, reward, and next state
                sender.Send(srd);
                sender.Close();

            } catch (ArgumentNullException ane) {
                print ("ArgumentNullException : " + ane.ToString ());
            } catch (SocketException se) {
                print ("SocketException : " + se.ToString ());
            } catch (Exception e) {
                print ("Unexpected exception : " +  e.ToString ());
            }

        } catch (Exception e) {
            print (e.ToString ());
        }


    }
	public float[] StringToArray(string input, string separator)
	{
	    string[] stringList = input.Split(separator.ToCharArray(), 
		                              StringSplitOptions.RemoveEmptyEntries);
	    float[] list = new float[stringList.Length];

	    for (int i = 0; i < stringList.Length; i++)
	    {
		// list[i] = Convert.ChangeType(stringList[i], typeof(float));
        list[i] = (float) Convert.ToDouble(stringList[i]);
	    }

	    return list;
	}

    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), UnityEngine.Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), UnityEngine.Random.Range(-10f, 10f));
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(UnityEngine.Random.Range(-1.5f, 1.5f), 4f, UnityEngine.Random.Range(-1.5f, 1.5f))
            + gameObject.transform.position;
        //Reset the parameters when the Agent is reset.

        step_count = 0;
        totalReward = 0;

        SetResetParameters();
    }

    public override float[] Heuristic()
    {
        var action = new float[2];

        action[0] = -Input.GetAxis("Horizontal");
        action[1] = Input.GetAxis("Vertical");
        return action;
    }

    public void SetBall()
    {
        //Set the attributes of the ball by fetching the information from the academy
        m_BallRb.mass = m_ResetParams["mass"];
        var scale = m_ResetParams["scale"];
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetBall();
    }
}



'''