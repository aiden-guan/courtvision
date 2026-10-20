"""Colours (BGR) used by the synthetic renderer and the colour-blob detector.

Shared contract: ``sim.render.SyntheticRenderer`` draws with these exact colours
and ``perception.detector.ColorBlobDetector`` segments them.
"""

FLOOR_BGR = (140, 180, 215)
LINE_BGR = (250, 250, 250)
PAINT_BGR = (110, 140, 190)
HOME_BGR = (40, 160, 40)  # green jerseys
AWAY_BGR = (150, 40, 30)  # blue jerseys
REF_BGR = (25, 25, 25)  # black referee shirts
BALL_BGR = (0, 110, 255)  # orange
RIM_BGR = (0, 0, 220)  # red rim
BACKBOARD_BGR = (235, 235, 235)
SKIN_BGR = (120, 150, 200)
NUMBER_BGR = (255, 255, 255)
