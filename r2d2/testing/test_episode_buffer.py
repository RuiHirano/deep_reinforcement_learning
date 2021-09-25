import sys
sys.path.append('./../')
import unittest
from lib.episode_buffer import EpisodeBuffer, Transition

class TestEpisodeBuffer(unittest.TestCase):
    def setUp(self):
        pass

    def test_modify_timesteps(self):
        buffer = EpisodeBuffer(4,4)
        [buffer.add(Transition(i, i, i, i, i, i, i, i)) for i in range(20)]
        segments = buffer.pull_segments()
        expected = 4
        self.assertEqual(expected, len(segments))

    def test_modify_timesteps_len_0(self):
        buffer = EpisodeBuffer(4,4)
        segments = buffer.pull_segments()
        expected = 0
        self.assertEqual(expected, len(segments))

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()