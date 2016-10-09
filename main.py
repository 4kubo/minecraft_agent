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
            <ServerInitialConditions>
                <Time>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <DefaultWorldGenerator/>
                <ServerQuitFromTimeUp timeLimitMs="300000"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Rover</Name>
            <AgentStart>
                <Inventory>
                    <InventoryBlock slot="0" type="sand" quantity="64"/>
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
    def __init__(self, option):
        self.opiton = option
        self.video_width = 296
        self.video_height = 160
        self.dimention = 2
        # controls how quickly the agent responds to getting stuck, and the amount of time it waits for on a "wait" command.
        self.cyclesPerCheck = 10
        self.state_list = [u'Yaw', u'Pitch']
        self.commandSequences = {i : command for i, command in enumerate(command_list)}
        self.agent_state_dict = {verb : 0.0 for verb
                                 in set(command.split(' ')[0] for command in command_list)}
        n_episode = 1000
        max_retries = 5
        num_state = len(self.state_list) + len(self.agent_state_dict)

        my_mission = MalmoPython.MissionSpec(GetMissionXML(self.video_width, self.video_height), True)

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy( MalmoPython.VideoPolicy.LATEST_FRAME_ONLY )


        agent = Agent(len(self.commandSequences), num_state, self.video_width, self.video_height,\
                      self.dimention)

        for episode in xrange(n_episode):
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
            print 'taotal reward at episode {1} : {0:.3}'.format(total_rewards, episode)


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
        world_state = self.preprocess(world_state)

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

        # initialization
        action = 1
        obvs_text = world_state.observations[-1].text
        observation = json.loads(obvs_text)
        agent.y_pos = observation[u'YPos']
        agent_state, obsv_image = self.get_agent_state(world_state, action)
        state_context = agent.get_initial_state(obsv_image)

        if hasattr(self, 'window'):
            QApplication.processEvents()

        while world_state.is_mission_running:
            world_state = self.preprocess(world_state)
            if world_state.number_of_observations_since_last_state > 0:
                currentCycle += 1
                if currentCycle == self.cyclesPerCheck:  # Time to check our speed and decrement our wait counter (if set):
                    currentCycle = 0
                    if waitCycles > 0:
                        waitCycles -= 1

            if waitCycles <= 0:
                agent_state, obsv_image = self.get_agent_state(world_state, action)
                reward = calc_reward(world_state, agent)
                action = agent.get_action(state_context, agent_state)
                state_context = agent.run(state_context, action, reward, obsv_image, agent_state)

                if agent.t % 100 == 0:
                    print 'total reward :', agent.total_reward

                command = self.commandSequences[action]
                verb,sep,param = command.partition(" ")
                if verb == "wait":  # "wait" isn't a Malmo command - it's just used here to pause execution of our "programme".
                    waitCycles = int(param.strip())
                else:
                    waitCycles = 1
                    self.agent_host.sendCommand(command)    # Send the command to Minecraft.

            if hasattr(self, 'window'):
                QApplication.processEvents()

        return agent.total_reward


    def preprocess(self, world_state):
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = self.agent_host.peekWorldState()

        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.agent_host.peekWorldState()
        world_state = self.agent_host.getWorldState()
        return world_state


    def get_state_array(self, action):
        command = self.commandSequences[action]
        verb, param = command.split(' ')
        self.agent_state_dict[verb] = float(param)
        if verb != "wait":
            self.agent_state_dict["wait"] = 0.0
        return self.agent_state_dict.values()


    def get_agent_state(self, world_state, action):
        obvs_text = world_state.observations[-1].text
        observation = json.loads(obvs_text)
        state_array = self.get_state_array(action)
        # one_hot vector
        state_array = state_array + [observation[attr] for attr in self.state_list]
        agent_state = np.array(state_array)

        # getting perceived image
        observation_img = world_state.video_frames[0].pixels
        # for monitoring
        if hasattr(self, 'window'):
            p_img = QImage(str(observation_img),  self.video_width, self.video_height, self.video_width*3, QImage.Format_RGB888)
            p_map = QPixmap.fromImage(p_img)
            self.label.setPixmap(p_map)
            self.label.update()

        if self.dimention is 3:
            # observation_img = np.array(Image.frombytes('RGB',\
            #                     (self.video_width, self.video_height), str(observation_img)))
            observation_img = np.array(observation_img, dtype=np.uint8)\
                            .reshape((self.video_width, self.video_height, 3))
            return observation_img.reshape(self.video_width, self.video_height, 3)
        elif self.dimention is 2:
            observation_img = np.array(Image.frombytes('RGB',\
                        (self.video_width, self.video_height), str(observation_img)).convert('L'))
            observation_img = observation_img.reshape(self.video_width, self.video_height)
            return agent_state, observation_img
        else:
            raise RuntimeError


def calc_reward(world_state, agent):
    obvs_text = world_state.observations[-1].text
    observation = json.loads(obvs_text)

    pos_reward = observation[u'YPos'] - agent.y_pos
    reward = pos_reward

    # updating agent state
    agent.y_pos = observation[u'YPos']
    return reward


if __name__ == '__main__':
    option = arg_parsing.arg_parse()
    if option.monitoring is True:
        app = QApplication([])
    main = Main(option)
