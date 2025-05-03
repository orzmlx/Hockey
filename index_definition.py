"""
define the events that are considered as strength events
"""

STRENGTH_EVENTS = [
    #The offensive incident in the opponent's half of the field
    "(eventname =='pass') &  (outcome == 'successful') & (type in ['ozentrystretch','d2doffboards','outletoffboards','rushoffboards','rush','eastwest']) ",
    "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 0)",
    "(eventname =='check') & (inopponentarea == 0) & (outcome == 'successful')",
    #   "(eventname =='dumpin') & (outcome =='successful')",
    #Any shot counts as an attack, whether it succeeds or fails.
    "(eventname =='shot') ",
    # Attempt to draw a foul in the opponent's half of the court.
    # "(eventname =='penaltydrawn' & outcome == 'successful')",# keep 迫使对手犯规获得多打少机会，提升后续进攻威胁
    # 对方半场争球成功
    "(eventname =='faceoff') & (type in ['recoveredwithshotonnet','recoveredwithshotonnetandslotshot','recoveredwithslotshot']) & (outcome == 'successful')",
    # keep
    # The opponent successfully defended the ball in their half of the field.
    "(eventname =='puckprotection') & (inopponentarea == 0) & (outcome == 'successful')",  # keep

    "(eventname =='controlledentry') & (outcome == 'successful')",  # keep

    "(eventname =='dumpin')  & (inopponentarea == 0) & (outcome == 'successful')",
    ]

"""
define the events that are considered as defend events

"""
# Successful in retaining possession in one's own half of the field.
DEFEND_EVENTS = {"(eventname =='block') & (outcome == 'successful')  & (inopponentarea == 1)",
                 # Success lies in successfully resolving one's own predicament
                 "(eventname =='dumpout')  & (inopponentarea == 1) & (outcome == 'successful')",
                 # Successful legal collision and ball recovery in the opponent's half of the field.
                 "(eventname =='check') & (inopponentarea == 1) & (outcome == 'successful')",
                 # Attempt to commit a foul in one's own half of the field
                 #    "(eventname =='penaltydrawn') & (inopponentarea == 1)",
                 "(eventname =='faceoff') & (inopponentarea == 1) & (outcome == 'successful')",
                 # Defensive ball control in one's own half of the field
                 "(eventname =='puckprotection') & (inopponentarea == 1) & (outcome == 'successful')",  # keep

                 # Successfully recovering the ball in one's own half of the field
                 # "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 1)",
                 # Successfully brought out
                 "(eventname =='controlledexit') & (outcome == 'successful')",

                 "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 1)",
                 # "(eventname =='controlledentryagainst') & (type in ['3on2','1on1','2on2','3on3','2on1','3on1']) ",
             }

#BODY ={"body_check":"eventname == 'check' & type=='body'","puckprotection":"eventname == 'puckprotection' & type=='body'"}
BODY = {"body": f"{'type'}.str.contains('{'body'}')"}

OPDUMP = {"opdump":f"{'type'}.str.contains('{'opdump'}')"}
STRETCH = {"stretch": "type == 'stretch'"}
RECOVER = {"recover": "type == 'recovered'"}
RUSH = {"rush": f"{'type'}.str.contains('{'rush'}')"}
THREE_ON_TWO = {"three_on_two": "type == '3on2'"}
TWO_ON_THREE = {"two_on_three": "type == '2on3'"}
CARRY = {"carry": "eventname == 'carry'"}
CONTEST = {"contest": f"{'type'}.str.contains('{'contest'}')"}
GOAL = {"goal":"eventname == 'goal'"}
SHOT = {"shot":"eventname == 'shot'"}
PASS = {"pass":"eventname == 'pass'"}
BLOCK = {"block":"eventname == 'block'"}
CHECK  = {"check":"eventname == 'check'"}
SAVE = {"save":"eventname == 'save'"}
PROTECTION = {"puckprotection":"eventname == 'puckprotection'"}
#Receiving the ball accurately in the opponent's half of the field
ACCURACY_RECEPTION = {"accuracy_reception":"eventname == 'reception'  &  inopponentarea == 0"}
#Precise Pass
ACCURACY_PASS = {"accuracy_pass":"eventname == 'pass'  & type in ['ozentrystretch','d2doffboards','outletoffboards','rushoffboards','rush','eastwest']"}
#Effective defense within one's own territory
EFFICIENT_BLOCK = {"efficient_block" : "eventname == 'block'  & inopponentarea == 1"}
#Successful Body Check to Intercept the Ball
BODY_CHECK = {"body_check":"eventname == 'check' & type=='body' "}
# Key Rescue Operation
# SAVE = "eventname == 'save' & type == 'none'"
#Successful ball control in our own half of the field
SELF_AREA_PROTECTION = {"self_area_protection":"(eventname =='check') & (inopponentarea == 1)"}

CONFIDENCE_INDEX = { **SHOT,**ACCURACY_RECEPTION, **ACCURACY_PASS, **EFFICIENT_BLOCK, **BODY_CHECK,**GOAL,
                     **SELF_AREA_PROTECTION}
EXERTION_INDEX = {**RUSH, **CONTEST, **BODY, **BLOCK, **SAVE, **PROTECTION, **CARRY,
                  **CHECK,  **PASS,**THREE_ON_TWO,**RECOVER,**STRETCH,**TWO_ON_THREE,**OPDUMP}

# Type	Direction	Usage	Technology	Risk Level
# north	Vertical	Quick Advance	Direct Pass	Medium
# eastwest	Horizontally Oriented	Organizing Offensive Play	Direct Pass	High
# d2d	Horizontally Oriented	Safe Control	Direct Pass	Low
# outletoffboards	Vertical	Defensive Zone Outgoing Ball Passing	Board Wall Passing	Medium
# ozentryoffboards	Vertical	Offensive Zone Entry	Board Wall Passing	Medium
# stretchoffboards	Vertical	Quick Counterattack	Board Wall + Long Pass	High
HIGH_RISK_PASS= ['eastwest','stretchoffboards']
MEDIUM_RISK_PASS = ['ozentryoffboards','outletoffboards','slot']
LOW_RISK_PASS = ['d2d']
# Key Rescue Operation # SAVE = "eventname == 'save' & type == 'none'"
# Own Halfway Line Ball Control Success # Classified by Tactical Purposes
# Major Category	Sub-type Included	Typical Scenarios
# Guarding Zone Outgoing Ball (Outlet, Outletoffboards, D2D Offboards)	Overcoming the opponent's forward-line pressure
# Offensive Zone Entry (Ozentry, Ozentry Offboards)	Establishing control in the offensive zone
# Rapid Counterattack (Stretch, Stretch Offboards, Rush)	Utilizing speed to break through the defense line
# High-Risk Organization (East-West, East-West Offboards)	Lateral transfer but prone to interception
# Safe Control (D2D, South Offboards)	Stable rhythm of passing between defenders
Defensive_zone_passing = ['outlet', 'outletoffboards', 'd2doffboards']
Entry_into_offensive_zone = ['ozentry', 'ozentryoffboards']
Quick_counterattack = ['stretch', 'stretchoffboards', 'rush']
High_risk_organization = ['eastwest', 'eastwestoffboards']
Safe_control = ['d2d', 'southoffboards']

PASS_STATISTIC = {'HIGH_RISK_PASS': HIGH_RISK_PASS,
                  'MEDIUM_RISK_PASS':MEDIUM_RISK_PASS,
                  'LOW_RISK_PASS':LOW_RISK_PASS,
                  'Defensive_zone_passing':Defensive_zone_passing,
                  'Entry_into_offensive_zone':Entry_into_offensive_zone,
                  'Quick_counterattack':Quick_counterattack,
                  'High_risk_organization':High_risk_organization,
                  'Safe_control':Safe_control}


Slingshot_chain = [['pass_south',  'pass_north'],

                   #['pass_south', 'reception_regular', 'pass_north', 'reception_regular']
                   ]
# Multiple returns, seeking offensive opportunities
Reset_Recycle_chain = ['pass_south', 'pass_south']



overcome_pressure_chain = [['pass_south', 'block_pass'],
                           ['pass_south', 'controlledexit_carrywithplay']]



attach_chain =[['pass_south',  'shot_outside'],# inefficiency actions

['pass_south',  'pass_slot'],

['pass_south',  'shot_slot']]



#create space
east_west_chain = [['pass_south', 'pass_eastwest']]


# Quick alternation implies that the opponent is aggressively pushing forward or is executing a tactic of exploiting space by pulling it apart.


pass_rapid_chain = [['pass_south', 'pass_north', 'pass_south'],
                    ['pass_south', 'pass_south', 'pass_south']]

pass_south_pattern = {'Slingshot_chain':Slingshot_chain,
                       'Reset_Recycle_chain' :  Reset_Recycle_chain,
                      'overcome_pressure_chain':overcome_pressure_chain,
                      'east_west_chain':east_west_chain,
                      'pass_rapid_chain':pass_rapid_chain}