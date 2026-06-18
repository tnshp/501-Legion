import os

if __name__ == "__main__":
    from kaggle_environments import make
    # os.chdir('./submission')
    # cwd = os.getcwd()

    for i in range(100):
        env = make("orbit_wars", debug= True)
        
        steps = env.run(["agent.py", "random"])
        # print(steps[-1][0].observation)
        terminal = steps[-1][0].reward
        if terminal == 1:
            print("won")
        else:
            print("lost")