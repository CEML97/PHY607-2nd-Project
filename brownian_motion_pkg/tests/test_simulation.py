import unittest
import numpy as np
from src.brownian_motion_sim import BrownianMotion

class TestBrownianMotion(unittest.TestCase):
    def setUp(self):
        self.bm = BrownianMotion(T=1, N=1000, x0=0, v0=1, L=100, p=0, 
                                gamma=0.5, m=1.0, sigma=1.0)

    def test_initialization(self):
        self.assertEqual(self.bm.T, 1)
        self.assertEqual(self.bm.N, 1000)
        self.assertEqual(self.bm.dt, 0.001)

    def test_simulation_methods(self):
        x_euler, v_euler = self.bm.run_simulation(method='euler')
        x_rk4, v_rk4 = self.bm.run_simulation(method='rk4')
        
        self.assertEqual(len(x_euler), 1000)
        self.assertEqual(len(x_rk4), 1000)