# -*- coding:utf-8 -*-
import MalmoPython
import sys
import time
import json
import numpy as np
from PIL import Image
from PyQt4.QtGui import QApplication, QWidget, QPixmap, QLabel, QImage
# import cv2
sys.path.append('/home/amamori/.local/lib/malmo/Malmo-0.17.0-Linux-Ubuntu-14.04-64bit/Python_Examples')

from my_agent import Agent


def GetMissionXML(width, height):
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
                    <Width>''' + str(width) + '''</Width>
                    <Height>''' + str(height) + '''</Height>
                </VideoProducer>
                <ContinuousMovementCommands/>
                <ObservationFromFullStats/>
                <ObservationFromRay/>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''

# Variety of strategies for dealing with loss of motion:
# commandSequences=[
#     "jump 1; move 1; wait 1; jump 0; move 1; wait 2",   # attempt to jump over obstacle
#     "turn 0.5; wait 1; turn 0; move 1; wait 2",         # turn right a little
#     "turn -0.5; wait 1; turn 0; move 1; wait 2",        # turn left a little
#     "move 0; attack 1; wait 5; pitch 0.5; wait 1; pitch 0; attack 1; wait 5; pitch -0.5; wait 1; pitch 0; attack 0; move 1; wait 2", # attempt to destroy some obstacles
#     "move 0; pitch 1; wait 2; pitch 0; use 1; jump 1; wait 6; use 0; jump 0; pitch -1; wait 1; pitch 0; wait 2; move 1; wait 2" # attempt to build tower under our feet
# ]

command_list=[
    "move 1", "move 0", "jump 1", "jump 0", "turn 0.5", "turn -0.5", "turn 0",
    "wait 1", "pitch 0.5", "pitch -0.5", "pitch 0", "jump 1", "jump 0", "use 1", "use 0",
    "attack 1", "attack 0"]


class Main:
    def __init__(self):
        self.video_width = 296
        self.video_height = 160
        self.dimention = 2

        self.commandSequences = {i : command for i, command in enumerate(command_list)}

        my_mission = MalmoPython.MissionSpec(GetMissionXML(self.video_width, self.video_height), True)

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setVideoPolicy( MalmoPython.VideoPolicy.LATEST_FRAME_ONLY )

        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print 'ERROR:',e
            print self.agent_host.getUsage()
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print self.agent_host.getUsage()
            exit(0)

        if self.agent_host.receivedArgument("test"):
            my_mission.timeLimitInSeconds(20) # else mission runs forever

        # Attempt to start the mission:
        print 'The agent is starting...'
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, MalmoPython.MissionRecordSpec() )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print "Error starting mission",e
                    print "Is the game running?"
                    exit(1)
                else:
                    time.sleep(2)

        # Wait for the mission to start:
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print '.',
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        # Prepare for monitoring
        self.window = QWidget()
        self.label = QLabel(self.window)
        self.label.setPixmap(QPixmap.grabWidget(self.window,\
                             0, 0, self.video_width, self.video_height))
        self.window.setGeometry(0, 0, self.video_width, self.video_height)
        self.window.resize(self.video_width, self.video_height)
        self.window.show()

        self.mainloop(world_state)


    def mainloop(self, world_state):
        # action = 0
        cyclesPerCheck = 10 # controls how quickly the agent responds to getting stuck, and the amount of time it waits for on a "wait" command.
        currentCycle = 0
        waitCycles = 1
        is_started = False

        # Main loop:
        print 'start!'

        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            # initialization
            if not is_started:
                while world_state.number_of_video_frames_since_last_state < 1:
                    time.sleep(0.1)
                    world_state = self.agent_host.getWorldState()
                raw_observation = world_state.video_frames[0].pixels
                observation = self.preprocess(raw_observation)
                agent = Agent(len(self.commandSequences), self.video_width, self.video_height,\
                              self.dimention)
                agent_state = agent.get_initial_state(observation)
                is_started = True

            if world_state.number_of_observations_since_last_state > 0:
                currentCycle += 1
                if currentCycle == cyclesPerCheck:  # Time to check our speed and decrement our wait counter (if set):
                    currentCycle = 0
                    if waitCycles > 0:
                        waitCycles -= 1

            if waitCycles <= 0:
                if world_state.number_of_observations_since_last_state > 0:
                    obvsText = world_state.observations[-1].text
                    observation = json.loads(obvsText)

                while len(world_state.video_frames) == 0:
                    world_state = self.agent_host.getWorldState()
                raw_observation = world_state.video_frames[0].pixels

                reward = calc_reward(observation)
                observation = self.preprocess(raw_observation)
                action = agent.get_action(agent_state)
                agent_state = agent.run(agent_state, action, reward, observation)

                command = self.commandSequences[action]
                print command
                verb,sep,param = command.partition(" ")
                if verb == "wait":  # "wait" isn't a Malmo command - it's just used here to pause execution of our "programme".
                    waitCycles = int(param.strip())
                else:
                    waitCycles = 1
                    self.agent_host.sendCommand(command)    # Send the command to Minecraft.

            QApplication.processEvents()


    def preprocess(self, observation):
        # image = np.array(Image.frombytes('RGB', (self.video_width, self.video_height), str(observation)))
        # cv2.imshow('observation', image)
        # cv2.waitKey(100)

        p_img = QImage(str(observation),  self.video_width, self.video_height, self.video_width*3, QImage.Format_RGB888)
        p_map = QPixmap.fromImage(p_img)
        self.label.setPixmap(p_map)
        self.label.update()

        if self.dimention is 3:
            # observation = np.array(Image.frombytes('RGB',\
            #                     (self.video_width, self.video_height), str(observation)))
            observation = np.array(observation, dtype=np.uint8)\
                            .reshape((self.video_width, self.video_height, 3))
            return observation.reshape(self.video_width, self.video_height, 3)
        elif self.dimention is 2:
            observation = np.array(Image.frombytes('RGB',\
                        (self.video_width, self.video_height), str(observation)).convert('L'))
            return observation.reshape(self.video_width, self.video_height)
        else:
            raise RuntimeError


def calc_reward(observation):
    return 1.0

# Mission has ended.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
