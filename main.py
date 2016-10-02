# -*- coding:utf-8 -*-
import MalmoPython
import sys
import time
import datetime
import json
import numpy as np
from PIL import Image
sys.path.append('/home/amamori/.local/lib/malmo/Malmo-0.17.0-Linux-Ubuntu-14.04-64bit/Python_Examples')

from my_agent import Agent

VIDEO_WIDTH = 296
VIDEO_HEIGHT = 160
DIMENTION = 2


def preprocess(observation, dimention):
    if dimention is 3:
        observation = np.array(Image.frombytes('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), str(observation)))
        return observation.reshape(VIDEO_WIDTH, VIDEO_HEIGHT, 3)
    elif dimention is 2:
        observation = np.array(Image.frombytes('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), str(observation)).convert('L'))
        return observation.reshape(VIDEO_WIDTH, VIDEO_HEIGHT)
    else:
        raise RuntimeError


def calc_reward(observation):
    return 1.0


def GetMissionXML():
    ''' Build an XML mission string that uses the DefaultWorldGenerator.'''

    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Normal life</Summary>
        </About>

        <ServerSection>
            <ServerHandlers>
                <DefaultWorldGenerator/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Rover</Name>
            <AgentStart>
                <Inventory>
                    <InventoryBlock slot="0" type="glowstone" quantity="63"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <VideoProducer>
                    <Width>''' + str(VIDEO_WIDTH) + '''</Width>
                    <Height>''' + str(VIDEO_HEIGHT) + '''</Height>
                </VideoProducer>
                <ContinuousMovementCommands/>
                <ObservationFromFullStats/>
                <ObservationFromRay/>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''

# Variety of strategies for dealing with loss of motion:
commandSequences=[
    "jump 1; move 1; wait 1; jump 0; move 1; wait 2",   # attempt to jump over obstacle
    "turn 0.5; wait 1; turn 0; move 1; wait 2",         # turn right a little
    "turn -0.5; wait 1; turn 0; move 1; wait 2",        # turn left a little
    "move 0; attack 1; wait 5; pitch 0.5; wait 1; pitch 0; attack 1; wait 5; pitch -0.5; wait 1; pitch 0; attack 0; move 1; wait 2", # attempt to destroy some obstacles
    "move 0; pitch 1; wait 2; pitch 0; use 1; jump 1; wait 6; use 0; jump 0; pitch -1; wait 1; pitch 0; wait 2; move 1; wait 2" # attempt to build tower under our feet
]
commandSequences = {i : command for i, command in enumerate(commandSequences)}

my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)

agent_host = MalmoPython.AgentHost()
agent_host.setVideoPolicy( MalmoPython.VideoPolicy.LATEST_FRAME_ONLY )

try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)

if agent_host.receivedArgument("test"):
    my_mission.timeLimitInSeconds(20) # else mission runs forever

# Attempt to start the mission:
print 'The agent is starting...'
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, MalmoPython.MissionRecordSpec() )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print "Error starting mission",e
            print "Is the game running?"
            exit(1)
        else:
            time.sleep(2)

# Wait for the mission to start:
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print '.',
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

currentSequence="move 1; wait 4"    # start off by moving
currentSpeed = 0.0
distTravelledAtLastCheck = 0.0
timeStampAtLastCheck = datetime.datetime.now()
cyclesPerCheck = 10 # controls how quickly the agent responds to getting stuck, and the amount of time it waits for on a "wait" command.
currentCycle = 0
waitCycles = 0

is_started = False

# Main loop:
print 'start!'

while world_state.is_mission_running:
    world_state = agent_host.getWorldState()
    # initialization
    if not is_started:
        while world_state.number_of_video_frames_since_last_state < 1:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        observation = world_state.video_frames[0].pixels
        observation = preprocess(observation, DIMENTION)
        agent = Agent(len(commandSequences), VIDEO_WIDTH, VIDEO_HEIGHT, DIMENTION)
        agent_state = agent.get_initial_state(observation)
        is_started = True

    if world_state.number_of_observations_since_last_state > 0:
        obvsText = world_state.observations[-1].text
        currentCycle += 1
        if currentCycle == cyclesPerCheck:  # Time to check our speed and decrement our wait counter (if set):
            currentCycle = 0
            if waitCycles > 0:
                waitCycles -= 1
            # Now use the latest observation to calculate our approximate speed:
            data = json.loads(obvsText) # observation comes in as a JSON string...
            dist = data.get(u'DistanceTravelled', 0)    #... containing a "DistanceTravelled" field (amongst other things).
            timestamp = world_state.observations[-1].timestamp  # timestamp arrives as a python DateTime object

            delta_dist = dist - distTravelledAtLastCheck
            delta_time = timestamp - timeStampAtLastCheck
            currentSpeed = 1000000.0 * delta_dist / float(delta_time.microseconds)  # "centimetres" per second?

            distTravelledAtLastCheck = dist
            timeStampAtLastCheck = timestamp

    if waitCycles == 0:
        # if state has been changed
        if world_state.number_of_observations_since_last_state > 0:
            obvsText = world_state.observations[-1].text
            observation = json.loads(obvsText)

        # Time to execute the next command, if we have one:
        if currentSequence != "":
            commands = currentSequence.split(";", 1)
            command = commands[0].strip()
            if len(commands) > 1:
                currentSequence = commands[1]
            else:
                currentSequence = ""
            print command
            verb,sep,param = command.partition(" ")
            if verb == "wait":  # "wait" isn't a Malmo command - it's just used here to pause execution of our "programme".
                waitCycles = int(param.strip())
            else:
                agent_host.sendCommand(command)    # Send the command to Minecraft.

    if currentSequence == "" and currentSpeed < 50 and waitCycles == 0\
       and observation is not None and u'LineOfSight' in observation:
        reward = calc_reward(observation)
        while world_state.number_of_video_frames_since_last_state < 1:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        observation = world_state.video_frames[0].pixels
        observation = preprocess(observation, DIMENTION)
        action = agent.get_action(agent_state)
        agent_state = agent.run(agent_state, action, reward, observation)
        currentSequence = commandSequences[action]
        data = None
        print "Stuck! Chosen programme: " + currentSequence

# Mission has ended.
