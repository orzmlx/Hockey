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
GOAL = {"goal":"eventname == 'goal'"}
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


CONFIDENCE_INDEX = {**GOAL, **ACCURACY_RECEPTION, **ACCURACY_PASS, **EFFICIENT_BLOCK, **BODY_CHECK}