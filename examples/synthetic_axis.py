"""Demonstrate geometric output only; this is not an enzyme structure or activity test."""
import json
import numpy as np

from mcsa_benchmark import load_case
from mcsa_benchmark.mechanisms.adenylate import angle, inside

nu = {'xyz': np.array([-3.2, 0.0, 0.0])}
center = {'xyz': np.array([0.0, 0.0, 0.0])}
leaving = {'xyz': np.array([1.6, 0.0, 0.0])}
limits = load_case('1ZIO')['mechanism_criteria']['cutoffs']
distance = float(np.linalg.norm(nu['xyz']-center['xyz']))
theta = angle(nu, center, leaving)
print(json.dumps({'synthetic': True, 'distance_A':distance, 'angle_deg':theta,
                  'axis_only_pass':inside(distance,limits['reaction_distance_A']) and inside(theta,limits['attack_angle_deg']),
                  'not_evaluated':['protein','Mg coordination','general checks','complete case']}))

