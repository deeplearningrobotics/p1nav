import unittest
import replay_buffer
import random

class ReplayBufferTest(unittest.TestCase):
    def test_sample(self):
        for i in random.sample(range(1, 100), 20):

            buffer = replay_buffer.ReplayBuffer(i)
            data1 = [self.get_rand_tuple() for k in range(i*2)]


            for sample in data1:
                buffer.add(*sample)

            samples = buffer.sample(i)
            expected = list(reversed(data1[-i:]))
            self.assertEqual(len(samples), len(expected))
            self.assertEqual(set(samples), set(expected))

    def get_rand_tuple(self):
        return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

if __name__ == '__main__':
    unittest.main()
