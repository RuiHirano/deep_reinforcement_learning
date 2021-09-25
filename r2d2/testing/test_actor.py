import sys
sys.path.append('./../')
import unittest
from lib.episode_buffer import EpisodeBuffer, Transition
from lib.actor import Actor, ActorParameter
import ray

class TestActor(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_multi_step_transition(self):
        pass

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()