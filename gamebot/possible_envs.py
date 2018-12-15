import gym
from gym import envs
print(envs.registry.all())
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)




env = gym.make('Walker2d-v2')# Creates the environment
env.reset()


for q in range(1000):
    env.render()
    env.step(env.action_space.sample())

# agent
# environment
# reward
# done
# info


# obs, reward, done, info = env.step()





