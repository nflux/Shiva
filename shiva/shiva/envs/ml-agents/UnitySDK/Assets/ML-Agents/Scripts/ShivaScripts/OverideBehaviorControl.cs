using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class OverideBehaviorControl : MonoBehaviour
{
    [SerializeField]
    BehaviorParameters [] param;
    [SerializeField]
    Agent []agent;
    public void Update()
    {
        if (Input.GetKey("c"))
        {
            for (int i = 0; i < param.Length; i++)
            {
                if( param[i] != null)
                {
                    if (param[i].Heuristic == false)
                        param[i].Heuristic = true;
                    else if (param[i].Heuristic == true)
                        param[i].Heuristic = false;
                }


            }
           
        }

        if (Input.GetKey("d"))
        {            
           for ( int i = 0; i < param.Length; i ++)
            {
                if(agent[i] != null)
                {
                    if (agent[i].agentParameters.onDemandDecision == false)
                        agent[i].agentParameters.onDemandDecision = true;
                    else if (agent[i].agentParameters.onDemandDecision == true)
                        agent[i].agentParameters.onDemandDecision = false;
                }
                
            }
            
        }
    }
} 
