import math
import numpy as np

class ObjState:
    def __init__(self, position, velocity):
        self.pos = position
        self.vel = velocity

class State:
    def __init__(self, num_agents, seed=29):
        target_pos = np.zeros(2)
        target_vel = np.zeros(2)
        self.target_state = ObjState(target_pos, target_vel)

        self.num_agents = num_agents
        np.random.seed(seed)
        rads = math.sqrt(2) * np.random.rand(num_agents)
        angles = (2.0 * math.pi / num_agents) * np.random.rand(num_agents)
        self.agent_states = []

        for i in range(num_agents):
            angles[i] += (i * 2.0 * math.pi) / num_agents
            pos_x = rads[i] * math.cos(angles[i])
            pos_y = rads[i] * math.sin(angles[i])
            agent_pos = np.asarray([pos_x, pos_y])
            agent_vel = np.zeros(2)
            agent_state = ObjState(agent_pos, agent_vel)
            self.agent_states.append(agent_state)

    def in_box(self):
        if (self.target_state.pos[0] ** 2 >= 1\
                or self.target_state.pos[1] ** 2 >= 1):
            return True
        else:
            return False

    def print_state(self):
        print("Target position:", self.target_state.pos)
        print("Target velocity:", self.target_state.vel)

        print("Agent info: ")
        for i in range(self.num_agents):
            print("Agent", i, "position:", self.agent_states[i].pos)
            print("Agent", i, "velocity:", self.agent_states[i].vel)

def main():
    test_state = State(3, 29)
    test_state.print_state()

if __name__ == '__main__':
    main()
