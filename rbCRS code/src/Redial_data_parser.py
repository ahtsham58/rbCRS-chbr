#Importing the modules
import pandas as pd
import numpy as np
import simplejson as json

data=[]

with open('../dialogs_data/redial_jason_data.txt', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

dialogs = data['foo']

parsed_dialogues = []

#Int&RS dialogs list
#seekerId_list =[956,959,961,960,960,961,961,961,961,960,960,960,960,961,960,959,959,960,967,963,969,960,959,959,971,960,960,960,959,973,959,979,979,980,959,982,959,972,972,972,976,986,976,959,959,959,988
   # ,967,972,972,972,990,967,991,991,992,960,993,993,994,995,976,995,997,998,997,997,1001,960,995]
#gtId_list =[957,958,960,961,961,960,960,960,960,961,962,962,963,964,964,964,965,965,966,968,959,970,960,960,960,959,959,959,960,972,979,959,959,981,979,959,983,984,984,984,985,976,985,987,976,976,976,973
    #,989,989,989,960,991,992,992,991,990,986,960,992,976,996,976,995,992,999,999,1000,992,960]

#Retrieval system dialogs list
seekerId_list =[368,510,433,403,629,183,591,366,14,562,37,608,245,340,190,627,926,406,743,789,462,4,360,599,479,502,196,492,423,690,842,177,599,144,315,375,619,299,832,608,751,433,409,1,637,
               99,747,710,440,767,18,615,366,287,254,35,183,810,832,457,239,14,203,275,275,365,261,641,563,365]
gtId_list =[171,793,492,365,177,360,562,339,21,547,29,619,75,335,269,629,383,385,747,781,451,254,254,426,505,500,401,552,250,665,72,629,153,32,401,368,611,550,903,586,907,461,366,220,675,
            4,391,758,315,792,1,707,319,317,366,710,341,794,743,743,315,53,39,251,251,423,550,706,547,385]

for key, d in enumerate(dialogs):
    tex_tmessages_raw = []
    messages = dialogs[key]['messages']
    seeker_text = ''
    gt_text = ''
    tex_tmessages_raw.append('CONVERSATION:' + str(key +1) +'\n')
    for msgid, msg in enumerate(messages):
        senderId = messages[msgid]['senderWorkerId']
        if senderId == seekerId_list[key]:
            if gt_text:
                tex_tmessages_raw.append('GROUND TRUTH: <s>' + gt_text + '</s> \n')
                gt_text = ''
                seeker_text =  seeker_text +' '+ messages[msgid]['text']
            else:
                seeker_text =  seeker_text +' ' + messages[msgid]['text']

        elif senderId == gtId_list[key]:
            if seeker_text:
                tex_tmessages_raw.append('SEEKER: <s>' + seeker_text + '</s> \n')
                seeker_text = ''
                gt_text = gt_text+' '  + messages[msgid]['text']
            else:
                gt_text = gt_text +' ' + messages[msgid]['text']

    if gt_text:
        tex_tmessages_raw.append('GROUND TRUTH: <s>' + gt_text + '</s> \n')
    elif seeker_text:
        tex_tmessages_raw.append('SEEKER: <s>' + seeker_text + '</s> \n')
    parsed_dialogues.append(tex_tmessages_raw)

with open('../dialogs_data/input_dialogs_study.txt', 'w') as filehandle:
        for dia in parsed_dialogues:
            filehandle.writelines("%s" % place for place in dia)

