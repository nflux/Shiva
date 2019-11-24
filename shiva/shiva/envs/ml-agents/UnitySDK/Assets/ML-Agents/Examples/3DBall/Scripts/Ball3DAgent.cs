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

        print("BrainZ: " + actionZ);
        print("BrainX: " + actionZ);

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

            // Connect the socket to the remote endpoint. Catch any errors.  
            try {
                sender.Connect (remoteEP);

                // print ("Socket connected to {0}" + sender.RemoteEndPoint.ToString ());

                // Encode the data string into a byte array.  
                byte[] obs1 = Encoding.ASCII.GetBytes (id.ToString() + " " + state);

                // print("Made the obs");
                // Send the data through the socket.  
                sender.Send (obs1);       // here's the first obs i send     
                // print("Sent the obs");

                // Receive the response from the remote device.  
                int bytesRec = sender.Receive(bytes);
                actions = Encoding.ASCII.GetString(bytes, 0, bytesRec);

                actionZ = 0.0f;//actions[0]; // Here we overwrite the Tensorflow Brain actions
                actionX = 0.0f;//actions[1]; 


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
                    Done();
                    SetReward(-1f);
                    AgentReset();

                } else {
                    done = false;
                    reward = 0.1f;
                    SetReward(0.1f);
                    
                }

                byte[] srd = Encoding.ASCII.GetBytes(
                    id.ToString() + " " +         // the id doesn't seem to matter at the moment
                    next_state + " " +
                    reward.ToString() + " " +
                    done.ToString()
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

    public override void AgentReset()
    {
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), UnityEngine.Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), UnityEngine.Random.Range(-10f, 10f));
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(UnityEngine.Random.Range(-1.5f, 1.5f), 4f, UnityEngine.Random.Range(-1.5f, 1.5f))
            + gameObject.transform.position;
        //Reset the parameters when the Agent is reset.
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
