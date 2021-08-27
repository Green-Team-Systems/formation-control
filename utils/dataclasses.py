from dataclasses import dataclass
import dataclasses


@dataclass
class PosVec3():
    """
    Coordinate vector that describes the x, y and z
    position of an agent in a specific reference
    frame, as given by the frame reference tag. The
    global coordinate frame is the world relative frame
    as defined in the Earth-Centered, Earth-Facing (ECEF)
    coordinate system.

    Starting positions can also be tracked via the
    """
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    frame: str = "local"
    starting: bool = False


@dataclass
class Quaternion():
    """
    Orientation-based structure to hold the values
    of the quaternion that defines the orientation
    of the vehicle at any time step t.
    """
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    W: float = 1.0