from env import MagnetsEnv
import tty, termios, sys

def getchar():
   # a sketchy getchar
   # https://gist.github.com/jasonrdsouza/1901709
   fd = sys.stdin.fileno()
   old_settings = termios.tcgetattr(fd)
   try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
      if (ch == '\x1b'):
          # do a bad hack to get an arrow key
          sys.stdin.read(1)
          ch = sys.stdin.read(1)
          if (ch == 'A'):
              ch = 'KEY_UP'
          elif (ch == 'C'):
              ch = 'KEY_RIGHT'
          elif (ch == 'D'):
              ch = 'KEY_LEFT'
          elif (ch == 'B'):
              ch = 'KEY_DOWN'
   finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
   return ch

key2action = {
    ' ': [8, 8, 8],

    'w': [6, 8, 8],
    'a': [4, 8, 8],
    's': [2, 8, 8],
    'd': [0, 8, 8],

    'i': [8, 6, 8],
    'j': [8, 4, 8],
    'k': [8, 2, 8],
    'l': [8, 0, 8],

    'KEY_UP':   [8, 8, 6],
    'KEY_LEFT': [8, 8, 4],
    'KEY_DOWN': [8, 8, 2],
    'KEY_RIGHT':[8, 8, 0],
}

if __name__ == '__main__':
    env = MagnetsEnv(num_agents=3)
    env.render()
    while (True):
        key = getchar()
        env.step(key2action[key])
        env.render()
