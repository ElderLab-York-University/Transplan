from Libs import *
from Utils import *

def AverageCountsMC(args, args_mc):
    estimates = []
    for arg in args_mc:
        with open(arg.CountingResPth) as f:
            estimates.append(json.load(f))

    averaged = {}
    for i in args_mc[0].MetaData["gt"]:
        data_i = []
        for estimate in estimates:
            if estimate[i] > 0: data_i.append(estimate[i])

        if len(data_i) > 0:
            averaged[i] = int(np.round(np.mean(data_i)))
        else:
            averaged[i] = 0
    
    
    with open(args.CountingResPth, "w") as f:
        json.dump(averaged, f, indent=2)

    return SucLog("averaged counts MC")

def IntegrateCounts(args, args_mc):
    estimates = []
    for arg in args_mc:
        with open(arg.CountingResPth) as f:
            estimates.append(json.load(f))

    averaged = {}
    for arg, estimate in zip(args_mc, estimates):
        if arg.SubID[-3:] == "sc1":
            for moi in ["1", "11", "12"]:
                averaged[moi] = estimate[moi] if moi in estimate else 0
        elif arg.SubID[-3:] == "sc2":
            for moi in ["4", "9", "10"]:
                averaged[moi] = estimate[moi] if moi in estimate else 0

        elif arg.SubID[-3:] == "sc3":
            for moi in ["3", "6", "7", "8"]:
                averaged[moi] = estimate[moi] if moi in estimate else 0

        elif arg.SubID[-3:] == "sc4":
            for moi in ["2", "5"]:
                averaged[moi] = estimate[moi] if moi in estimate else 0
    
    with open(args.CountingResPth, "w") as f:
        json.dump(averaged, f, indent=2)

    return SucLog("integrated counts MC")
