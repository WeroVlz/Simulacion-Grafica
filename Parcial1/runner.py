import simpy
import random


class Runner(object):

    def __init__(self, _env: simpy.Environment, resource: simpy.Resource, _id: int) -> None:
        self.total_distance = 0.0
        self.env = _env
        self.total_boost = 0.0
        self.total_rest = 0.0
        self.id = _id
        self.res = resource
        self.action = self.env.process(self.run())

    def run(self) -> None:
        while True:
            try:
                with self.res.request() as req:
                    result = yield req
                    if result:
                        print("The runner %d started running at %2f" % (self.id, self.env.now))
                        energy_boost = abs(random.normalvariate(5))
                        # print("boost: %.2f" % energy_boost)
                        self.total_boost += energy_boost
                        yield self.env.timeout(energy_boost)
                        if random.random() < 0.3:
                            self.total_boost += energy_boost * 300
                        else:
                            self.total_distance += energy_boost * 600

                print("the runner %d got tired and now is walking at %2f" % (self.id, self.env.now))
                recovery_time = abs(random.normalvariate(2))
                # print("recovery: %.2f" % recovery_time)
                self.total_rest += recovery_time
                yield self.env.timeout(recovery_time)
                self.total_distance += recovery_time * 100
            except simpy.Interrupt:
                print("Runner %d got a cramp at %.2f" % (self.id, self.env.now))


def cramp_generator(_env: simpy.Environment, runner: Runner) -> None:
    if random.random() < 0.35:
        print("Runner %d will get a cramp" % runner.id)
        yield _env.timeout(abs(random.normalvariate(12)))
        runner.action.interrupt()


env = simpy.Environment()

res = simpy.Resource(env, 1)
runner1 = Runner(env, res, 1)
runner2 = Runner(env, res, 2)
env.process(cramp_generator(env, runner1))
env.process(cramp_generator(env, runner2))

env.run(until=20)

print("The runner %d went for %d meters" % (runner1.id, runner1.total_distance))
print("The runner %d went for %d meters" % (runner2.id, runner2.total_distance))
# print("Total boost: %.2f, Total rest: %.2f" % (runner1.total_boost, runner1.total_distance))
