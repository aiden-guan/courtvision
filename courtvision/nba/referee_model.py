"""The referee classifier's weights (``nba/referee.py``), written by tools/referee_classifier.py: do not
edit by hand. Fitted on human labels of these NBA games (Roboflow Universe "basketball-player-detection-2",
CC BY 4.0; research use): boston-celtics-new-york-knicks-game-1, boston-celtics-new-york-knicks-game-2, boston-celtics-vs-orlando-magic-game-1, golden-state-warriors-houston-rockets-game-3, los-angeles-clippers-vs-denver-nuggets-game-1, los-angeles-lakers-vs-minesota-timberwolves-game-1, oklahoma-city-thunder-denver-nuggets-game-1, oklahoma-city-thunder-denver-nuggets-game-3.
Each game left out in turn: referees 5753/6101, players called referee 1730/22440 (the shirt rule
4509/6101 and 2464), F1 0.847 against 0.690.
"""

FEATURES = ('shirt_s_median', 'shirt_v_median', 'shirt_v_p10', 'shirt_v_p20', 'shirt_v_p90', 'shirt_dark', 'shirt_bright_grey', 'shirt_coloured', 'shirt_grad_x', 'shirt_grad_y', 'shirt_hue_0', 'shirt_hue_1', 'shirt_hue_2', 'shirt_hue_3', 'shirt_hue_4', 'shirt_hue_5', 'trousers_s_median', 'trousers_v_median', 'trousers_v_p10', 'trousers_v_p20', 'trousers_v_p90', 'trousers_dark', 'trousers_bright_grey', 'trousers_coloured')
MEAN = (0.2827079797, 0.5198052521, 0.2865283659, 0.3450914436, 0.7725497524, 0.1632438578, 0.22891078, 0.4092738036, 0.05726215353, 0.04627650107, 0.2130158026, 0.00898940414, 0.03273872501, 0.03410282764, 0.04555796711, 0.07486907716, 0.3107535459, 0.5676265049, 0.330997362, 0.400723378, 0.7875049083, 0.1983412832, 0.2708508948, 0.4422626292,)
STD = (0.1295716416, 0.2067775838, 0.1554999372, 0.1801324639, 0.1681834879, 0.1859523667, 0.2287647848, 0.241873294, 0.01936578648, 0.01537771183, 0.2023887712, 0.03218468659, 0.1031530163, 0.08561350212, 0.08337072799, 0.1251647548, 0.1374405668, 0.2633123791, 0.1975853361, 0.2275070411, 0.2038329111, 0.2855644797, 0.2598450093, 0.267190041,)
COEF = (-0.3852306976, 0.6857993919, -0.6407362244, -0.5328718143, 0.6526053501, 0.1341652332, -0.628200529, -0.2547850482, -0.7787384631, -0.2934187415, -0.142563811, -0.1725155876, 0.01268692526, -0.2233203544, -0.2965137771, 0.1223274737, 0.06744956658, -0.2104854915, -0.6125233728, -0.4962727357, 0.1506627386, 1.531181092, -0.2058229904, -0.2200701706,)
BIAS = -2.104895693
THRESHOLD = 0.5
