"""
define the events that are considered as strength events
"""
# 对方半场的进攻事件
STRENGTH_EVENTS = [
    "(eventname =='pass') &  (outcome == 'successful') & (type in ['ozentrystretch','d2doffboards','outletoffboards','rushoffboards','rush','eastwest']) ",
    "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 0)",
    "(eventname =='check') & (inopponentarea == 0) & (outcome == 'successful')",
    #   "(eventname =='dumpin') & (outcome =='successful')",
    # 只要是shot都算进攻，无论成功或者失败
    "(eventname =='shot') ",
    # 尝试在对方半场造犯规
    # "(eventname =='penaltydrawn' & outcome == 'successful')",# keep 迫使对手犯规获得多打少机会，提升后续进攻威胁
    # 对方半场争球成功
    "(eventname =='faceoff') & (type in ['recoveredwithshotonnet','recoveredwithshotonnetandslotshot','recoveredwithslotshot']) & (outcome == 'successful')",
    # keep
    # 对方半场护球成功
    "(eventname =='puckprotection') & (inopponentarea == 0) & (outcome == 'successful')",  # keep

    "(eventname =='controlledentry') & (outcome == 'successful')",  # keep

    "(eventname =='dumpin')  & (inopponentarea == 0) & (outcome == 'successful')",
    ]

"""
define the events that are considered as defend events

"""  # 成功在己方半场护球成功
DEFEND_EVENTS = {"(eventname =='block') & (outcome == 'successful')  & (inopponentarea == 1)",
                 # 成功在己方解围成功
                 "(eventname =='dumpout')  & (inopponentarea == 1) & (outcome == 'successful')",
                 # 成功在己方半场合法撞击抢球
                 "(eventname =='check') & (inopponentarea == 1) & (outcome == 'successful')",
                 # # 尝试在己方半场造犯规
                 #    "(eventname =='penaltydrawn') & (inopponentarea == 1)",
                 "(eventname =='faceoff') & (inopponentarea == 1) & (outcome == 'successful')",
                 # 己方半场护球
                 "(eventname =='puckprotection') & (inopponentarea == 1) & (outcome == 'successful')",  # keep

                 # 在己方半场抢球成功
                 # "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 1)",
                 # 成功带出
                 "(eventname =='controlledexit') & (outcome == 'successful')",

                 "(eventname =='lpr') & (outcome == 'successful') & (inopponentarea == 1)",
                 # "(eventname =='controlledentryagainst') & (type in ['3on2','1on1','2on2','3on3','2on1','3on1']) ",
             }
#进球
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
#对方半场精准接球
ACCURACY_RECEPTION = {"accuracy_reception":"eventname == 'reception'  &  inopponentarea == 0"}
#精准传球
ACCURACY_PASS = {"accuracy_pass":"eventname == 'pass'  & type in ['ozentrystretch','d2doffboards','outletoffboards','rushoffboards','rush','eastwest']"}
#己方区域的有效防守
EFFICIENT_BLOCK = {"efficient_block" : "eventname == 'block'  & inopponentarea == 1"}
#身体对抗成功断球
BODY_CHECK = {"body_check":"eventname == 'check' & type=='body' "}
#关键扑救
# SAVE = "eventname == 'save' & type == 'none'"
#己方半场护球成功
SELF_AREA_PROTECTION = {"self_area_protection":"(eventname =='check') & (inopponentarea == 1)"}

CONFIDENCE_INDEX = { **SHOT,**ACCURACY_RECEPTION, **ACCURACY_PASS, **EFFICIENT_BLOCK, **BODY_CHECK,**GOAL,
                     **SELF_AREA_PROTECTION}
EXERTION_INDEX = {**RUSH, **CONTEST, **BODY, **BLOCK, **SAVE, **PROTECTION, **CARRY,
                  **CHECK,  **PASS,**THREE_ON_TWO,**RECOVER,**STRETCH,**TWO_ON_THREE,**OPDUMP}

# 类型	方向	用途	技术	风险等级
# north	纵向	快速推进	直接传球	中
# eastwest	横向	组织进攻	直接传球	高
# d2d	横向	安全控制	直接传球	低
# outletoffboards	纵向	守区出球	板墙传球	中
# ozentryoffboards	纵向	攻区进入	板墙传球	中
# stretchoffboards	纵向	快速反击	板墙+长传	高
HIGH_RISK_PASS= ['eastwest','stretchoffboards']
MEDIUM_RISK_PASS = ['ozentryoffboards','outletoffboards','slot']
LOW_RISK_PASS = ['d2d']
# 按战术用途分类
# 大类	包含的具体类型	典型场景
# 守区出球	outlet, outletoffboards, d2doffboards	破解对手前场压迫
# 攻区进入	ozentry, ozentryoffboards	建立攻区控球
# 快速反击	stretch, stretchoffboards, rush	利用速度突破防线
# 高风险组织	eastwest, eastwestoffboards	横向转移但易被拦截
# 安全控制	d2d, southoffboards	后卫间传递稳定节奏
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
                   #传球失败，就没有下面这个链
                   #['pass_south', 'reception_regular', 'pass_north', 'reception_regular']
                   ]
#多次回传，找进进攻机会
Reset_Recycle_chain = ['pass_south', 'pass_south']



overcome_pressure_chain = [['pass_south', 'block_pass'],
                           ['pass_south', 'controlledexit_carrywithplay']]



attach_chain =[['pass_south',  'shot_outside'],# 低效率

['pass_south',  'pass_slot'],

['pass_south',  'shot_slot']]



#create space
east_west_chain = [['pass_south', 'pass_eastwest']]



# 快速交替，意味着对手前压激烈，或者执行拉扯空间的战术意图


pass_rapid_chain = [['pass_south', 'pass_north', 'pass_south'],
                    ['pass_south', 'pass_south', 'pass_south']]

pass_south_pattern = {'Slingshot_chain':Slingshot_chain,
                       'Reset_Recycle_chain' :  Reset_Recycle_chain,
                      'overcome_pressure_chain':overcome_pressure_chain,
                      'east_west_chain':east_west_chain,
                      'pass_rapid_chain':pass_rapid_chain}