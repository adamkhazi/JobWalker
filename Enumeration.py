from enum import Enum

class App_State():
    Unknown = -1
    Event = 0
    Applied = 1
    Interviewed = 2
    Offered = 3

class Test_Mode(Enum):
    Random_Candidate = 1
    Warm_Candidate = 0
    Cold_Candidate = 2
    Cold_Opportunity = 3