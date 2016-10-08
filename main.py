# -*- coding:utf-8 -*-
import MalmoPython
import sys
import time
import json
import numpy as np
from PIL import Image
from PyQt4.QtGui import QApplication, QWidget, QPixmap, QLabel, QImage
sys.path.append('/home/amamori/.local/lib/malmo/Malmo-0.17.0-Linux-Ubuntu-14.04-64bit/Python_Examples')

from my_agent import Agent
import arg_parsing

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

command_list=[
    "move 1", "move 0", "jump 1", "jump 0", "turn 0.5", "turn -0.5", "turn 0",
    "wait 1", "pitch 0.5", "pitch -0.5", "pitch 0", "jump 1", "jump 0", "use 1", "use 0",
    "attack 1", "attack 0"]


class Main:
    def __init__(self):
        option = arg_parsing.arg_parse()

        max_retries = 1000

        self.video_width = 296
        self.video_height = 160
        self.dimention = 2
        # controls how quickly the agent responds to getting stuck, and the amount of time it waits for on a "wait" command.
        self.cyclesPerCheck = 10


        self.commandSequences = {i : command for i, command in enumerate(command_list)}

        my_mission = MalmoPython.MissionSpec(GetMissionXML(self.video_width, self.video_height), True)

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy( MalmoPython.VideoPolicy.LATEST_FRAME_ONLY )


        agent = Agent(len(self.commandSequences), self.video_width, self.video_height,\
                      self.dimention)

        for episode in xrange(max_retries):
            # Attempt to start the mission:
            print 'The agent is starting...'
            max_retries = 3
            for retry in xrange(max_retries):
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
                sys.stdout.write('.')
                time.sleep(0.1)
                world_state = self.agent_host.getWorldState()
            print

            # Prepare for monitoring
            if option.monitoring is True:
                print 'Setting up monitor'
                self.setup_monitor()

            # Main loop
            total_rewards = self.mainloop(agent)
            print total_rewards


    def setup_monitor(self):
        if not hasattr(self, 'window'):
            self.window = QWidget()
            self.label = QLabel(self.window)
            self.label.setPixmap(QPixmap.grabWidget(self.window,\
                                 0, 0, self.video_width, self.video_height))
            self.window.setGeometry(0, 0, self.video_width, self.video_height)
            self.window.resize(self.video_width, self.video_height)
            self.window.show()


    def mainloop(self, agent):
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = self.agent_host.peekWorldState()

        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.agent_host.peekWorldState()
        world_state = self.agent_host.getWorldState()
        for err in world_state.errors:
            print err

        if not world_state.is_mission_running:
            print 'Mission has already ended'
            return 0

        assert len(world_state.video_frames) > 0, 'No video frames!?'

        currentCycle = 0
        waitCycles = 1

        # main loop:
        print 'Start!'

        # nitialization
        while world_state.number_of_video_frames_since_last_state < 1\
              or len(world_state.video_frames) is 0:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        raw_observation = world_state.video_frames[0].pixels
        observation = self.preprocess(raw_observation)
        agent_state = agent.get_initial_state(observation)
        QApplication.processEvents()

        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                currentCycle += 1
                if currentCycle == self.cyclesPerCheck:  # Time to check our speed and decrement our wait counter (if set):
                    currentCycle = 0
                    if waitCycles > 0:
                        waitCycles -= 1

            if waitCycles <= 0:
                while world_state.number_of_observations_since_last_state < 1\
                   or len(world_state.video_frames) == 0:
                    time.sleep(0.01)
                    world_state = self.agent_host.getWorldState()

                obvsText = world_state.observations[-1].text
                observation = json.loads(obvsText)
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
        if hasattr(self, 'window'):
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


if __name__ == '__main__':
    app = QApplication([])
    main = Main()
